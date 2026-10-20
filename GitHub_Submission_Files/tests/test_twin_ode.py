import unittest

from twin.glucose_ode import ModelParameters, State, simulate, step
from twin.kalman import ExtendedGlucoseFilter
from twin.personalization import fit_person_parameters


class GlucoseODETests(unittest.TestCase):
    def test_no_meal_baseline_is_steady(self):
        p=ModelParameters()
        states=simulate(110.0,minutes=240,baseline_mg_dl=110.0,p=p)
        self.assertAlmostEqual(states[-1].glucose_mg_dl,110.0,places=8)

    def test_carbohydrate_mass_balance(self):
        states=simulate(100.0,minutes=180,meal=60.0,baseline_mg_dl=100.0)
        final=states[-1]
        self.assertAlmostEqual(final.gut_fast_g+final.gut_slow_g+final.absorbed_g,60.0,places=5)

    def test_more_carbohydrate_has_monotonic_glucose_response(self):
        base=simulate(100.0,minutes=120,baseline_mg_dl=100.0)
        small=simulate(100.0,minutes=120,meal=20.0,baseline_mg_dl=100.0)
        large=simulate(100.0,minutes=120,meal=60.0,baseline_mg_dl=100.0)
        self.assertGreater(small[-1].glucose_mg_dl,base[-1].glucose_mg_dl)
        self.assertGreater(large[-1].glucose_mg_dl,small[-1].glucose_mg_dl)

    def test_parameter_bounds_are_enforced(self):
        with self.assertRaises(ValueError):
            ModelParameters(insulin_sensitivity=9.0)

    def test_kalman_update_moves_state_toward_measurement(self):
        filt=ExtendedGlucoseFilter(State(100.0),100.0,ModelParameters())
        filt.update(150.0)
        self.assertGreater(filt.state.glucose_mg_dl,100.0)
        self.assertLess(filt.state.glucose_mg_dl,150.0)

    def test_forecast_is_ordered_and_has_uncertainty_band(self):
        filt=ExtendedGlucoseFilter(State(120.0),100.0,ModelParameters())
        forecasts=filt.forecast_multiple((30,60,120))
        self.assertEqual(set(forecasts),{30,60,120})
        for forecast in forecasts.values():
            self.assertLessEqual(forecast["lower_95_mg_dl"],forecast["mean_mg_dl"])
            self.assertGreaterEqual(forecast["upper_95_mg_dl"],forecast["mean_mg_dl"])

    def test_personal_fit_falls_back_with_too_little_early_data(self):
        prior=ModelParameters(insulin_sensitivity=0.7)
        fit=fit_person_parameters([100,101],[0,0],[0,0],baseline_mg_dl=100,prior=prior)
        self.assertFalse(fit.stable)
        self.assertEqual(fit.params,prior)


if __name__=="__main__": unittest.main()
