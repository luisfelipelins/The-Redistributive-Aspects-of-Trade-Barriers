'''Short numerical checks; run with python code/test_nested_ces.py.'''
import contextlib
import io
import tempfile
import json
import runpy
import config
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

import functions
import functions_transition
import GMM
from ces_production import production,firm_at_interest,factor_demands
from GeneralEquilibriumModel import TypeModelParameters,TypeCalibParameters,GeneralEquilibriumModel


class NestedCESTests(unittest.TestCase):
    def setUp(self):
        self.par = TypeModelParameters(α            = 0.81,
                                       γ            = 0.55,
                                       ψ            = 0.40,
                                       χ            = -0.49,
                                       β_eff        = 1.1,
                                       τ            = 0.024,
                                       w_star       = 0.56,
                                       θ            = 1.0,
                                       σ            = 2.0,
                                       δ            = 0.95,
                                       ϱ            = 0.91,
                                       σ_ϵ          = 0.23,
                                       π_LL         = 0.99,
                                       π_HH         = 0.98,
                                       M            = 1,
                                       ξ            = 0,
                                       rebate_share = 1)
        self.cal = TypeCalibParameters(3,2,0,0,20,80,1e-6,20,1e-3,1e-3,0.03,1e-7,1e-5,0.001)

    def test_marginal_products_and_euler_identity(self):
        inputs = np.array([1.3,0.7,4.2])
        sol    = production(*inputs,self.par)
        prices = np.array([sol['pL'],sol['s'],sol['r']])
        for j in range(3):
            step     = inputs[j] * 1e-5
            up,down  = inputs.copy(),inputs.copy()
            up[j]   += step
            down[j] -= step
            derivative = (production(*up,self.par)['Y'] - production(*down,self.par)['Y']) / (2 * step)
            assert_allclose(derivative,prices[j],rtol=1e-8)
        assert_allclose(prices @ inputs,sol['Y'],rtol=1e-12)

    def test_cobb_douglas_limit(self):
        par = replace(self.par,ψ=0,χ=0)
        sol = production(1.3,0.7,4.2,par)
        Y   = 1.3**(1 - par.γ) * 0.7**(par.γ * (1 - par.α)) * 4.2**(par.γ * par.α)
        assert_allclose(sol['Y'],Y,rtol=1e-12)
        assert_allclose(sol['r'] * 4.2 / Y,par.γ * par.α,rtol=1e-12)

    def test_steady_transition_duality_and_tariff_balance(self):
        for theta in [1e-6,1.0,10.0]:
            par = replace(self.par,θ=theta)
            mod = GeneralEquilibriumModel(par,self.cal,log_dir=None)
            sol = firm_at_interest(0.269,0.03,mod.H,mod.L,par)
            w   = sol['pL'] / functions.Omega(0.269,par)
            par.w_star = w / (par.β * functions.t(0.269,par))
            with contextlib.redirect_stdout(io.StringIO()):
                mod.inner_loop_solver(0.03)
            trans = functions_transition.solve_transition_system_wrapper(mod.K_demand,par.β,mod.H,mod.L,par)
            assert_allclose([mod.I,trans['I'],trans['r']],[0.269,0.269,0.03],rtol=1e-7)
            assert_allclose(functions.solve_firm_side(mod.w,0.03,par)[2],mod.s,rtol=1e-7)
            demands = factor_demands(mod.w,mod.s,0.03,mod.I,mod.Y,par)
            assert_allclose(demands,[mod.L / (1 - mod.I),mod.H,mod.K_demand],rtol=1e-7)
            foreign_payment = mod.Y - mod.w * mod.L - mod.s * mod.H - 0.03 * mod.K_demand
            assert_allclose(mod.R,par.τ / (1 + par.τ) * foreign_payment,rtol=1e-7)
            assert_allclose(mod.state_probs @ mod.transfer,mod.R,rtol=1e-12)

    def test_wage_decomposition(self):
        K,H,L = 4.2,0.7,1.3
        sol   = production(L / (1 - 0.269),H,K,self.par)
        self.par.w_star = sol['pL'] / (functions.Omega(0.269,self.par) * self.par.β * functions.t(0.269,self.par))
        base  = functions_transition.solve_transition_system_wrapper(K,self.par.β,H,L,self.par)
        step  = 1e-5
        shock = functions_transition.solve_transition_system_wrapper(K * (1 + step),self.par.β * (1 + step),H,L,self.par)
        dI    = (shock['I'] - base['I']) / (1 - base['I'])
        dK    = np.log(shock['K'] / base['K'])
        dΩ    = np.log(shock['Ω'] / base['Ω'])
        pred  = -dΩ - (1 - self.par.ψ) * base['omega_X'] * dI
        pred += (1 - self.par.ψ) * base['omega_X'] * base['eta_K'] * dK
        self.assertLess(abs(np.log(shock['w'] / base['w']) - pred),1e-9)

    def test_four_gmm_parameters_and_moments(self):
        data = dict(I=0.269,w_to_wstar=1.47,skill_premium=1.85,low_skill_share=0.31)
        with patch.object(GMM,'GeneralEquilibriumModel') as model_class:
            model_class.return_value.outer_res = {'fun': 0.0}
            model_class.return_value.economy_statistics.return_value = dict(I=0.25,w_to_wstar=1.5,skill_premium=2.0,low_skill_share=0.28,low_skill_share_ni=0.33)
            residual = GMM.gmm_model_moments([2.0,0.2,1.1,0.62],self.par,self.cal,data,None,'unused.log')
            actual   = model_class.call_args.args[0]
            assert_allclose([actual.θ,actual.w_star,actual.β_eff,actual.γ],[2.0,0.2,1.1,0.62])
            assert_allclose([actual.α,actual.ψ,actual.χ],[0.81,0.40,-0.49])
            assert_allclose(residual,(np.array([0.269,1.47,1.85,0.31]) - [0.25,1.5,2.0,0.28]) / [0.269,1.47,1.85,0.31])

    def test_representative_agent_narrative_coefficient(self):
        par = self.par
        r   = 1 / par.δ - 1
        sol = firm_at_interest(0.269,r,1 / 3,2 / 3,par)
        par.w_star = sol['pL'] / (functions.Omega(0.269,par) * par.β * functions.t(0.269,par))
        base  = functions.representative_household_wrapper(par)
        shock = functions.representative_household_wrapper(replace(par,β_eff=par.β_eff * (1 + 1e-5)))
        Lambda = ((1 - par.ψ) * base['omega_X'] * (1 - par.χ) * base['eta_H']
                  / ((1 - par.χ) * base['eta_H'] + (1 - par.ψ) * base['omega_L'] * base['eta_K']))
        dΩ   = np.log(functions.Omega(shock['I'],par) / functions.Omega(base['I'],par))
        pred = -dΩ - Lambda * (shock['I'] - base['I']) / (1 - base['I'])
        self.assertLess(abs(np.log(shock['w'] / base['w']) - pred),1e-9)

    def test_legacy_resume_rejected_without_changing_log(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'gmm_run.log'
            path.write_text('Legacy Cobb-Douglas run\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Cobb-Douglas'):
                GMM.run_gmm(self.par,self.cal,{},np.eye(4),bounds=[(1,2)] * 4,resume_from=Path(folder))
            self.assertEqual(path.read_text(encoding='utf-8'),'Legacy Cobb-Douglas run\n')


    def test_fixed_gamma_history_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'gmm_run.log'
            text = 'Production: nested_ces\n'
            path.write_text(text,encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'did not estimate γ'):
                GMM.run_gmm(self.par,self.cal,{},np.eye(4),bounds=[(1,2)] * 4,resume_from=Path(folder))
            self.assertEqual(path.read_text(encoding='utf-8'),text)

    def test_four_parameter_history(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'gmm_run.log').write_text('Production: nested_ces\nEstimated parameters: θ,w_star,β_eff,γ\n',encoding='utf-8')
            params = [2.0,0.2,1.1,0.62]
            GMM._append_run_log(path / 'gmm_run.log',1,params,0.123)
            summary = 'α=0.81  γ=0.62  β_eff=1.1  w_star=0.2\n  θ=2.0  σ=2.0\n'
            (path / 'run_summary_00001.log').write_text(summary,encoding='utf-8')
            P,E = GMM._parse_run_history(path)
            assert_allclose(P,[params])
            assert_allclose(E,[0.123])

    def test_estimation_runner_saves_estimated_gamma(self):
        pre = json.loads((config.DATA_PARAMS / 'pre_gmm_params_ces.json').read_text(encoding='utf-8'))
        par = pre['parameters']
        H_to_L = pre['moments']['skill_premium'] * (1 - par['π_LL']) / (1 - par['π_HH'])
        expected_share = (1 - pre['moments']['capital_income_share']) / (1 + H_to_L)
        assert_allclose(pre['moments']['LS_share'],expected_share)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'pre_gmm_params_ces.json').write_text(json.dumps(pre),encoding='utf-8')
            best = dict(params=np.array([2.0,0.2,1.1,0.62]),obj=0.123)
            with patch.object(config,'DATA_PARAMS',path),patch.object(GMM,'run_gmm',return_value=(None,best)) as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    runpy.run_path(str(Path(__file__).with_name('gmm_estimator.py')))
            post = json.loads((path / 'post_gmm_params_ces.json').read_text(encoding='utf-8'))
            self.assertEqual(post['parameters']['γ'],0.62)
            self.assertEqual(run.call_args.kwargs['W'].shape,(4,4))
            self.assertEqual(run.call_args.kwargs['bounds'][-1],(0.01,0.99))
            self.assertEqual(run.call_args.kwargs['data_moments']['low_skill_share'],pre['moments']['LS_share'])


if __name__ == '__main__':
    unittest.main()
