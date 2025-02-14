if model_name in ["catboost", "merf"]:
            from catboost import CatBoostRegressor, CatBoostClassifier
            loss_function = "MAPE" if model_type == "REGRESSION" else "MultiClass"
            bootstrap_type = "Bernoulli" if model_type == "CLASSIFICATION" else "MVS"
            hyperparams = {
                "depth": 6,
                "subsample": 1.0,
                "bootstrap_type": bootstrap_type,
                "random_strength": 0.5,
                "reg_lambda": 0.1,
                "loss_function": loss_function,
                "early_stopping_rounds": 20,
                "random_seed": seed,
                "verbose": False,
            }
            if model_name == "catboost":
                model_cls = CatBoostRegressor if model_type == "REGRESSION" else CatBoostClassifier
                model = model_cls(**hyperparams, cat_features=cat_features)
            elif model_name == "merf":
                from merf import MERF
                hyperparams["iterations"] = 1000
                regr_cls = CatBoostRegressor if model_type == "REGRESSION" else CatBoostClassifier
                regr = regr_cls(**hyperparams, cat_features=cat_features)
                model = MERF(regr, max_iterations=10)




NEW

12:54
elif model_name == "ydf":
            import ydf
            templates = ydf.GradientBoostedTreesLearner.hyperparameter_templates()
            # A_K_
            #Replaceing the target_column here
            target_col  = 'Yield'
            cols = [col for col in X_train.columns if 'NDVI' in col or 'ESI' in col]
            features = []
            for col in cols:
                features.append(ydf.Feature(col,monotonic=+1))
            model = ydf.GradientBoostedTreesLearner(
                label=target_col,
                task=ydf.Task.REGRESSION,
                            features = features,
                            include_all_columns = True,
                            use_hessian_gain  = True,
                growing_strategy='LOCAL',
                categorical_algorithm='RANDOM',
                split_axis='SPARSE_OBLIQUE',
                sparse_oblique_normalization='MIN_MAX',
                sparse_oblique_num_projections_exponent=2.0)
            hyperparams = templates["benchmark_rank1v1"]
12:54
print("In MGWR")
            from mgwr.gwr import GWR , MGWR
            from mgwr.sel_bw import Sel_BW
            X_train = X_train.drop(columns = cat_features )
            coords = np.array(list(zip(X_train['lon'], X_train['lat'])))
            X_train = X_train.drop(columns = ['lon','lat']).values
            y_train = y_train.values.reshape((-1,1))
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            print(coords.shape,X_train.shape,y_train.shape)
            import warnings
            import scipy
            warnings.simplefilter("error")
            warnings.simplefilter("error", scipy.linalg.LinAlgWarning)  # For linear algebra warnings
            warnings.simplefilter("error", RuntimeWarning)
            for bw in range(len(y_train)-1 , 1,-1):
                try:
                    model = GWR(coords, y_train, X_train, bw=bw, fixed=False, kernel='bisquare')
                    results = model.fit()
                    break
                except ValueError:
                    pass
            if(bw<=1):
                raise Exception("No valid value for bw found.")
            results = model.fit()
            warnings.simplefilter("default")
            '''
            #X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
            #y_train = (y_train - y_train.mean(axis=0)) / y_train.std(axis=0)
            y_train = y_train.values.reshape((-1,1))
            selector = Sel_BW(coords, y_train, X_train,kernel='gaussian',fixed=True,multi=True)
            selector.search()
            model = MGWR(coords, y_train, X_train, selector, fixed=True, kernel='gaussian')
            '''
            '''
            try:
                selector = Sel_BW(coords, y_train, X_train, multi=True)
                selector.search(multi_bw_min=[2],multi_bw_max=[len(y_train)-1], search_memthod='interval')
            except:
                print(type(coords))
                print(type(y_train))
                print(type(X_train))
                selector = Sel_BW(coords, y_train, X_train, multi=False)
                selector.search(bw_min=[2],bw_max=[len(y_train)-1],search_method = 'interval')
            '''
            #model = MGWR(coords, y_train, X_train, selector, fixed=False, kernel='bisquare',sigma2_v1=True)
            #results = model.fit()
            print(results.R2)
            print("###########################")
            hyperparams = {}
            return hyperparams , model