import numpy as np
import optuna
from optuna.samplers import TPESampler

def fit_and_predict_with_GP(hp_constraints, X_train, y_train, X_test):
    '''Return predictions on surrogate model from GP.'''
    # NOTE: X_train, X_test are hyperparameter configurations, y_train is the corresponding validation error

    try:
        from smac.model.gaussian_process import GaussianProcess
    except ImportError:
        # Fallback for SMAC v1.x
        from smac.epm.gaussian_process import GaussianProcess
    from ConfigSpace import ConfigurationSpace, Float, Integer
    try:
        # In SMAC 1.4.0, these are often just scikit-learn kernels 
        # that SMAC expects to be passed into its GaussianProcess wrapper
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, Product
        # Rename them to match what the LLAMBO code expects (Kernel vs just Name)
        for kernel_class in [ConstantKernel, Matern, Product]:
            if not hasattr(kernel_class, 'prior'):
                kernel_class.prior = None
        MaternKernel = Matern
        ProductKernel = Product
    except ImportError:
        # If LLAMBO specifically needs the SMAC-wrapped versions
        from smac.epm.gaussian_process import ConstantKernel, MaternKernel, ProductKernel
    from sklearn.preprocessing import StandardScaler

    X_train = X_train.copy()
    X_test = X_test.copy()
    y_train = y_train.copy()

    # The ConfigSpace doesn't matter here, we just need to create a ConfigurationSpace object to use the GP
    # https://automl.github.io/SMAC3/main/_modules/smac/model/gaussian_process/gaussian_process.html#GaussianProcess
    cs = ConfigurationSpace(seed=42)
    for hp_name, hp_info in hp_constraints.items():
        type, transform, [hp_min, hp_max] = hp_info
        if type == 'int':
            if transform == 'log':
                cs.add_hyperparameter(Integer(hp_name, (hp_min, hp_max), log=True))
            else:
                cs.add_hyperparameter(Integer(hp_name, (hp_min, hp_max), log=False))
        elif type == 'float':
            if transform == 'log':
                cs.add_hyperparameter(Float(hp_name, (hp_min, hp_max), log=True))
            else:
                cs.add_hyperparameter(Float(hp_name, (hp_min, hp_max), log=False))

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()

    # standardize X_train, X_test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # kernel is based on default used here: 
    # https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html
    # length_scale = 1 since we have standardized the data, \nu=2.5 is good heuristic starting point
    kernel = ProductKernel(ConstantKernel(), MaternKernel(length_scale=1.0, nu=2.5))

    # Build types and bounds from ConfigSpace (required by SMAC GaussianProcess)
    types = []
    bounds = []
    for hp in cs.get_hyperparameters():
        if hasattr(hp, 'choices'):  # Categorical
            types.append(len(hp.choices))
            bounds.append((0, len(hp.choices) - 1))
        else:  # Float or Integer
            types.append(0)
            bounds.append((hp.lower, hp.upper))

    types = np.array(types, dtype=np.int32)
    bounds = np.array(bounds, dtype=np.float64)

    # SMAC GaussianProcess uses n_opt_restarts (not n_restarts)
    n_opt_restarts = 20 * int(X_train.shape[0] / 10) if X_train.shape[0] > 10 else 10
    gp = GaussianProcess(
        configspace=cs,
        types=types,
        bounds=bounds,
        seed=0,
        kernel=kernel,
        normalize_y=False,
        n_opt_restarts=n_opt_restarts,
    )
    gp.train(X_train, y_train)

    # Get predictions; SMAC GP API varies by version
    try:
        result = gp.predict(X_test)
        if isinstance(result, tuple) and len(result) == 2:
            y_pred, y_var = result
            y_std = np.sqrt(y_var)
        else:
            y_pred = result
            y_std = np.zeros_like(y_pred)
    except TypeError:
        try:
            y_pred, y_var = gp.predict(X_test, return_var=True)
            y_std = np.sqrt(y_var)
        except TypeError:
            y_pred, y_var = gp.predict_marginalized(X_test)
            y_std = np.sqrt(y_var)

    # Final cleanup to ensure arrays are the right shape
    y_pred = np.array(y_pred).flatten()
    y_std = np.array(y_std).flatten()

    return y_pred, y_std

def fit_and_predict_with_SMAC(hp_constraints, X_train, y_train, X_test):
    '''Return predictions on surrogate model from SMAC.'''

    try:
        # 1. Try SMAC v2.x path
        from smac.model.random_forest import RandomForest
    except ImportError:
        try:
            # 2. Try SMAC 1.4.x specific path (Most likely for you)
            from smac.epm.random_forest.rf_with_instances import RandomForestWithInstances as RandomForest
        except ImportError:
            try:
                # 3. Try legacy SMAC path
                from smac.epm.rf_with_instances import RandomForestWithInstances as RandomForest
            except ImportError:
                # 4. Fallback: Check if it's just in the epm folder as 'random_forest'
                from smac.epm.random_forest import RandomForest
    from ConfigSpace import ConfigurationSpace, Float, Integer

    config_space_dict = {}

    X_train = X_train.copy()
    X_test = X_test.copy()
    y_train = y_train.copy()

    cs = ConfigurationSpace(seed=42)
    for hp_name, hp_info in hp_constraints.items():
        type, transform, [hp_min, hp_max] = hp_info
        if type == 'int':
            if transform == 'log':
                cs.add_hyperparameter(Integer(hp_name, (hp_min, hp_max), log=True))
            else:
                cs.add_hyperparameter(Integer(hp_name, (hp_min, hp_max), log=False))
        elif type == 'float':
            if transform == 'log':
                cs.add_hyperparameter(Float(hp_name, (hp_min, hp_max), log=True))
            else:
                cs.add_hyperparameter(Float(hp_name, (hp_min, hp_max), log=False))

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()

    # SMAC v2 uses RandomForest(configspace, seed=...); SMAC 1.4 uses RandomForest(configspace, types, bounds, seed=...)
    try:
        if X_train.shape[0] < 5:
            rf = RandomForest(cs, seed=0, min_samples_leaf=1, min_samples_split=1)
        else:
            rf = RandomForest(cs, seed=0)
    except TypeError as e:
        if 'types' in str(e) and 'bounds' in str(e):
            # SMAC 1.4 API: need types and bounds from configspace
            try:
                from smac.utils.configspace import get_types as smac_get_types
                types, bounds = smac_get_types(cs)
            except (ImportError, AttributeError):
                try:
                    from smac.epm.utils import get_types as smac_get_types
                    types, bounds = smac_get_types(cs)
                except (ImportError, AttributeError):
                    # Fallback: build types/bounds for Integer and Float only (unit hypercube)
                    types, bounds = [], []
                    for hp in cs.get_hyperparameters():
                        types.append(0)  # continuous in normalized space
                        bounds.append((0.0, 1.0))
            if X_train.shape[0] < 5:
                rf = RandomForest(cs, types, bounds, seed=0, min_samples_leaf=1, min_samples_split=1)
            else:
                rf = RandomForest(cs, types, bounds, seed=0)
        else:
            raise

    rf.train(X_train, y_train)
    try:
        y_pred, y_var = rf.predict(X_test, covariance_type='diagonal')
    except TypeError:
        # SMAC 1.4 uses cov_return_type instead of covariance_type
        y_pred, y_var = rf.predict(X_test, cov_return_type='diagonal_cov')
    y_std = np.sqrt(y_var)

    return y_pred, y_std




def custom_gamma(x: int) -> int:
    return min(int(np.ceil(0.25 * x)), 25)


def fit_and_predict_with_TPE(hp_constraints, X_train, y_train, X_test, top_pct, multivariate, lower_is_better):
    '''Sample from independent TPE sampler'''

    def custom_gamma(x: int) -> int:
        return min(int(np.ceil(top_pct * x)), 25)

    direction = 'minimize' if lower_is_better else 'maximize'
    sampler =  TPESampler(multivariate=multivariate, # doing independent sampling if False, else multivariate
                          consider_prior=True,
                          consider_endpoints=False, # clip to hyp range
                          n_startup_trials=0,
                          gamma=custom_gamma, # custom gamma
                          n_ei_candidates=1,
                          seed=69)
    
    study = optuna.create_study(direction=direction, sampler=sampler)

    # create distribution to sample from
    distribution = {}
    for hp_name, hp_info in hp_constraints.items():
        type, transform, [hp_min, hp_max] = hp_info
        assert transform in ['log', 'logit', 'linear']
        if type == 'int':
            if transform == ['log']:
                distribution[hp_name] = optuna.distributions.IntDistribution(hp_min, hp_max, log=True)
            else:
                distribution[hp_name] = optuna.distributions.IntDistribution(hp_min, hp_max, log=False)
        elif type == 'float':
            if transform in ['log']:
                distribution[hp_name] = optuna.distributions.FloatDistribution(hp_min, hp_max, log=True)
            else:
                distribution[hp_name] = optuna.distributions.FloatDistribution(hp_min, hp_max, log=False)

    # add observed trials
    for i in range(len(X_train)):
        study.add_trial(
            optuna.trial.create_trial(
                params=X_train.iloc[i].to_dict(),
                distributions=distribution,
                value=y_train.iloc[i].values[0]
            )
        )

    rel_search_space = sampler.infer_relative_search_space(study, None)

    X_test = X_test.to_dict(orient='records')
    X_test_ = {k: [] for k in X_test[0].keys()}
    for d in X_test:
        for k, v in d.items():
            X_test_[k].append(v)
            
    for k, v in X_test_.items():
        X_test_[k] = np.array(v)

    if multivariate:
        score = sampler._evaluate_relative(study, None, rel_search_space, X_test_)
    else:
        score = sampler.evaluate_independent(study, distribution, X_test_)

    return score




