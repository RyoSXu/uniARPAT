"""The midpoint candidate changes order, retaining LR2 E+S and historical recipes."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from model.transformer import Transformer
from tests.data.test_spectral_balance import protocol
from tests.model.test_local_global_encoder import fixture
from tests.pipeline import test_spectral_residual as residual_fixtures
from tests.pipeline.test_global_norm import model_params
from tests.pipeline.test_local_structure_controls import predict
from tools.checks.check_long_budget import captured_config
from tools.eval.lg_valid_export import load_run
from tools.eval.local_global_compare import training_command
from tools.eval.periodic_geometry_acceptance import e3_operations
from tools.train import build_arg_parser
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.crystal import CrystalGeometry
from utils.local_global_experiment import capture_training_rng, comparison_contract, initialize_comparison_model
from utils.periodic_geometry import build_periodic_records
from utils.spectral_balance import ENERGY_METRIC, reconstruct_balanced_spectra
from utils.spectral_balance_experiment import validate_balance_config
from utils.spectral_layout_experiment import MIDPOINT_SCHEMA, layout_ablation_contract, validate_midpoint_model
from utils.zproj_alignment import state_hash


class SpectralLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def configuration(self, layout='midpoint'):
        return replace(residual_fixtures.SpectralResidualTests().configuration(),
                       local_layout=layout, tag='_unit_midpoint_es')

    def params(self, layout='midpoint'):
        return {**model_params('LR2'), 'scale_mode': 'per_atom', 'local_layout': layout}

    def create(self, layout='midpoint', dtype=torch.float32):
        params = self.params(layout)
        model = Transformer(**params).to(dtype).eval()
        info = initialize_comparison_model(model, params, 42)
        return model, info

    def contract(self, layout='midpoint'):
        with patch('utils.spectral_balance_experiment.validate_balance_config', return_value=protocol()):
            with patch('utils.spectral_balance_experiment.file_sha256', return_value='unit'):
                return comparison_contract(self.configuration(layout), self.params(layout))

    def test_identical_initialization_rng_and_parameter_layout_and_actual_order(self):
        reference, old_info = self.create('staged')
        candidate = Transformer(**self.params()).eval()
        before = torch.get_rng_state().clone()
        info = initialize_comparison_model(candidate, self.params(), 42)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(old_info['state_sha256'], info['state_sha256'])
        self.assertEqual(old_info['a64_initialization'], info['a64_initialization'])
        self.assertEqual(reference.encoder.update_counts, candidate.encoder.update_counts)
        self.assertEqual(sum(p.numel() for p in reference.parameters()), sum(p.numel() for p in candidate.parameters()))
        validate_midpoint_model(candidate, self.params(), require_identity=True)
        z, geometry = fixture()
        actual, expected = predict(candidate, z, geometry), predict(reference, z, geometry)
        self.assertTrue(any(not torch.equal(actual[key], expected[key]) for key in expected))
        candidate.encoder.layout = 'staged'
        restored = predict(candidate, z, geometry)
        for key in expected:
            torch.testing.assert_close(restored[key], expected[key], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            validate_midpoint_model(candidate, self.params())

    def test_forward_order_is_second_local_after_third_global_with_matching_gradients(self):
        for dtype, tolerance in ((torch.float32, 4e-5), (torch.float64, 2e-11)):
            model, _ = self.create(dtype=dtype)
            z, geometry = fixture(dtype)
            mask = z.eq(0)
            records = build_periodic_records(geometry, mask, 6.)
            trace, saved = [], {}
            handles = []
            for kind, layers in (('local', model.encoder.local_layers), ('global', model.encoder.global_layers)):
                for index, layer in enumerate(layers):
                    def record(_module, args, output, kind=kind, index=index):
                        trace.append((kind, index))
                        if kind == 'local' and index == 1:
                            saved['second_local_input'] = args[0].detach().clone()
                        if kind == 'global' and index == 2:
                            saved['third_global_output'] = output.detach().clone()
                    handles.append(layer.register_forward_hook(record))
            a = torch.randn((*z.shape, model.d_model), dtype=dtype, requires_grad=True)
            b = a.detach().clone().requires_grad_()
            try:
                actual = model.encoder(a, records, mask)
            finally:
                for handle in handles:
                    handle.remove()
            self.assertEqual(trace, [('local', 0), ('global', 0), ('global', 1), ('global', 2),
                                     ('local', 1), ('global', 3), ('global', 4), ('global', 5)])
            torch.testing.assert_close(saved['second_local_input'], saved['third_global_output'], rtol=0, atol=0)
            expected = model.encoder.local_layers[0](b.masked_fill(mask[..., None], 0), records, mask)
            for layer in model.encoder.global_layers[:3]:
                expected = layer(expected, mask)
            expected = model.encoder.local_layers[1](expected, records, mask)
            for layer in model.encoder.global_layers[3:]:
                expected = layer(expected, mask)
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
            weights = torch.randn_like(actual)
            params = tuple(model.encoder.parameters())
            left = torch.autograd.grad((actual*weights).sum(), (a, *params))
            right = torch.autograd.grad((expected*weights).sum(), (b, *params))
            for x, y in zip(left, right):
                self.assertTrue(torch.isfinite(x).all())
                torch.testing.assert_close(x, y, rtol=tolerance, atol=tolerance)

    def test_config_scope_cli_and_production_assembly_change_only_layout(self):
        cfg = self.configuration()
        policy = protocol((cfg.edos_num, cfg.phdos_num))
        for task, width in (('edos', cfg.delta_edos), ('phdos', cfg.delta_phdos)):
            policy['tasks'][task]['bin_width'] = width
        with patch('utils.spectral_balance_experiment.load_balance_protocol', return_value=policy):
            validate_balance_config(cfg)
            for changes in ({'local_route':'I3'}, {'spectral_shape_aux':False}, {'split_decoder':True},
                            {'phonon_head':True}, {'local_depth':3}, {'global_depth':5}, {'epochs':36},
                            {'use_amp':True}, {'spectral_balance_config':''}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    validate_balance_config(replace(cfg, **changes))
        with patch('utils.spectral_balance_experiment.validate_balance_config', return_value=protocol()):
            baseline = captured_config(self.configuration('staged'))
            changed = captured_config(cfg)
        expected = deepcopy(baseline)
        expected['model']['params']['sub_model']['transformer']['local_layout'] = 'midpoint'
        self.assertEqual(changed, expected)
        parsed = build_arg_parser().parse_args(training_command(cfg)[2:])
        self.assertEqual(parsed.local_layout, 'midpoint')
        for changes in ({'scale_mode':'eta'}, {'local_route':'I3'}, {'local_depth':1}, {'global_depth':5}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Transformer(**{**self.params(), **changes})

    def test_same_shape_checkpoints_are_isolated_and_same_contract_restores(self):
        old, new = self.contract('staged'), self.contract()
        self.assertEqual(new['schema'], MIDPOINT_SCHEMA)
        self.assertEqual(new['ablation'], layout_ablation_contract())
        for key in ('shape_supervision','spectral_balance','checkpoint_selection','backbone_reference'):
            self.assertEqual(old[key], new[key])
        model, _ = self.create()
        z, geometry = fixture()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        sum(value.square().mean() for value in predict(model,z,geometry).values()).backward()
        optimizer.step()
        checkpoint = deepcopy(build_ablation_checkpoint(1,'M1',42,False,model,optimizer,.3,
            model_contract=new,training_rng=capture_training_rng()))
        target, _ = self.create()
        target_optimizer = torch.optim.AdamW(target.parameters(), lr=5e-5)
        info = restore_ablation_checkpoint(checkpoint,target,target_optimizer,False,model_contract=new)
        self.assertTrue(info['optimizer_restored'])
        self.assertEqual(state_hash(model.state_dict()),state_hash(target.state_dict()))
        for index, slots in optimizer.state_dict()['state'].items():
            for key, value in slots.items():
                torch.testing.assert_close(target_optimizer.state_dict()['state'][index][key], value, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError,'different model contract'):
            restore_ablation_checkpoint(checkpoint,target,target_optimizer,False,model_contract=old)

    def test_export_rejects_wrong_order_layout_or_schema(self):
        contract = self.contract()
        saved = {'model_contract':contract, 'config':{'model':{'params':{
            'loss_form':ENERGY_METRIC,'spectral_balance':protocol(),'spectral_shape_aux':True,
            'sub_model':{'transformer':contract['transformer']}}}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config_used.yaml'
            path.write_text(yaml.safe_dump(saved))
            load_run(Path(directory))
            for field in ('schema','order','training_layout','model_layout','local_formula'):
                changed = deepcopy(saved)
                c = changed['model_contract']
                if field == 'schema':
                    c['schema'] = 'joint-spectral-balance-residual-model-v1'
                elif field == 'order':
                    c['ablation']['execution_order'][1] = ['local',1]
                elif field == 'training_layout':
                    c['training']['local_layout'] = 'staged'
                elif field == 'model_layout':
                    c['transformer']['local_layout'] = 'staged'
                else:
                    c['ablation']['local_formula'] = 'LN64(s+F(s,R))'
                path.write_text(yaml.safe_dump(changed))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    load_run(Path(directory))

    def test_physical_outputs_preserve_e3_and_atom_permutation(self):
        for dtype, tolerance in ((torch.float32,1e-4),(torch.float64,1e-9)):
            model, _ = self.create(dtype=dtype)
            z, geometry = fixture(dtype)
            def physical(zz, gg):
                outputs = predict(model,zz,gg)
                return {**outputs,**reconstruct_balanced_spectra(outputs,zz.ne(0).sum(-1),protocol((8,4)))}
            with torch.no_grad():
                expected = physical(z,geometry)
                for _, matrix, translation in e3_operations():
                    q,t = torch.as_tensor(matrix,dtype=dtype),torch.as_tensor(translation,dtype=dtype)
                    cell = geometry.cell@q
                    coords = geometry.frac_coords+torch.bmm(t.expand(2,1,3),torch.linalg.inv(cell))
                    actual = physical(z,CrystalGeometry(cell,coords))
                    for key in expected:
                        torch.testing.assert_close(actual[key],expected[key],rtol=tolerance,atol=tolerance)
                order = torch.tensor([2,0,3,1])
                actual = physical(z[:,order],CrystalGeometry(geometry.cell,geometry.frac_coords[:,order]))
                for key in expected:
                    torch.testing.assert_close(actual[key],expected[key],rtol=tolerance,atol=tolerance)


if __name__ == '__main__':
    unittest.main()
