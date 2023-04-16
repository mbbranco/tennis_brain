from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Import scores
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score


# Import classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import pandas as pd
import numpy as np
from datetime import datetime,date

from data_prep import data_import,get_more_info

class TennisPredModel():
    def __init__(self,p1_name,p2_name,matches,players_dict,tourn_info,match_round):
        self.p1_name = p1_name
        self.p2_name = p2_name

        self.p1_id = players_dict[p1_name][0]
        self.p2_id = players_dict[p2_name][0]

        self.p1_rank = players_dict[p1_name][1]
        self.p2_rank = players_dict[p2_name][1]

        self.p1_ratio = players_dict[p1_name][2]
        self.p2_ratio = players_dict[p2_name][2]

        self.players = {}
        self.players[p1_name] = self.p1_id
        self.players[p2_name] = self.p2_id

        self.matches = matches

        self.surface = tourn_info[0]
        #self.date = tourn_info[2]
        self.tourn_date = datetime.today().strftime("%Y-%m-%d")
        self.tourn_points = tourn_info[1]
        self.tourn_name = tourn_info[3]

        self.match_round = match_round

    def build_data_model(self):   
        features = ['tourney_date','tourney_name','winner_id','loser_id','winner_rank','loser_rank',
                    'winner_win_loss_ratio','loser_win_loss_ratio','tourney_points','surface','round_level']
     
        # prepare match to preodict
        inputs_match = [self.tourn_date, self.tourn_name, self.p1_id, self.p2_id, self.p1_rank, self.p2_rank, 
                        self.p1_ratio,self.p2_ratio,self.tourn_points, self.surface,self.match_round]
        
        df_input_match = pd.DataFrame([inputs_match],columns=features)
        df_input_match['h2h'] = 1

        
        # create dataset with all matches + match to predict
        matches_player = self.matches[features].copy()

        condition_1 = (matches_player['winner_id']== self.p1_id) & (matches_player['loser_id']== self.p2_id)
        condition_2 = (matches_player['winner_id']== self.p2_id) & (matches_player['loser_id']== self.p1_id)

        matches_player['h2h'] = np.where(((condition_1) | (condition_2)),1,0)

        matches_final = pd.concat([matches_player,df_input_match])
        matches_final = matches_final.reset_index(drop=True)
        dataset = matches_final.reset_index()
        self.dataset = dataset

    def prep_features(self):
        self.dataset['old'] = (datetime.now() - pd.to_datetime(self.dataset['tourney_date'],format='%Y-%m-%d')).dt.days
        type_dummy = pd.get_dummies(self.dataset['surface'])

        encoder = LabelEncoder()
        self.dataset['tourney_name_enc'] = encoder.fit_transform(self.dataset['tourney_name'])

        features_to_scale = ['winner_rank','loser_rank','winner_win_loss_ratio','loser_win_loss_ratio',
                             'tourney_points','round_level','tourney_name_enc','old']
        
        scl = StandardScaler()
        matches_scaled = scl.fit_transform(self.dataset[features_to_scale])
        matches_scaled_df = pd.DataFrame(matches_scaled, columns=features_to_scale)

        goal = pd.DataFrame(np.where((self.dataset['winner_id']==self.p1_id),1,0),columns=['result'])
        features_not_to_scale = ['index','h2h']
        matches_not_scaled_df = self.dataset[features_not_to_scale]

        final_df = pd.concat([matches_scaled_df,type_dummy,goal,matches_not_scaled_df],axis=1)
        self.dataset_scaled = final_df

    def prep_model(self,train_size_val=0.6,random_state_val=10):
        match_to_predict = pd.DataFrame(self.dataset_scaled.iloc[-1]).T
        matches_ready = self.dataset_scaled.iloc[0:-1]

        def create_feature_target_var(df):
            # Create feature variable
            X = df.drop('result', axis=1)
            # Create target variable
            y = df['result']
            return X,y
        
        X_to_predict,y_to_predict = create_feature_target_var(match_to_predict)

        X,y = create_feature_target_var(matches_ready)
        
        # Create training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size_val,random_state=random_state_val)

        return X, y, X_train, X_test, y_train, y_test, X_to_predict, y_to_predict


    def hyperparameter_tuning(self,name,X,y,X_train,y_train):
        if name=='LogisticRegression':
            # Logistic Regression Classifier
            parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
            model = LogisticRegression()
            log_cv = GridSearchCV(model, parameters, cv=5)
            log_cv.fit(X_train, y_train)
            model = LogisticRegression(C=log_cv.best_params_['C'])
            print(f'model {model}: {log_cv.best_params_}')

        elif name =='KNN':
            # KNN - Neighrest Neighbor 
            param_grid = {'n_neighbors': np.arange(1,50)}
            knn = KNeighborsClassifier()
            knn_cv = GridSearchCV(knn, param_grid, cv=3)
            knn_cv.fit(X,y)
            model = KNeighborsClassifier(n_neighbors=knn_cv.best_params_['n_neighbors'])
            print(f'model {model}: {knn_cv.best_params_}')

        elif name == 'SVC':
            parameters = {'C':[1, 10, 100]}
            svc = SVC()
            cv = GridSearchCV(svc, parameters, cv = 5)
            cv.fit(X, y)
            model = SVC(C=cv.best_params_['C'], probability=True)
            print(f'model {model}: {cv.best_params_}')
    
    def predictive_model(self,model,X,y,X_train,y_train,X_test,y_test):
        # Fit to the training data
        model.fit(X_train,y_train)

        # Compute accuracy
        accuracy = model.score(X_test,y_test)
        # print(f'Accuracy: {accuracy:.0%}')

        # Predict the labels of the test set
        y_pred = model.predict(X_test)

        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        # print(f'Precision: {precision:.0%}')
        # print(f'Recall: {recall:.0%}')

        # Generate the probabilities
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_pred_prob)
        f1_score_val = f1_score(y_test,y_pred)
        # print(f'AUC: {auc:.0%}')
        # print(f'F1 Score: {f1_score_val:.0%}')

        # # Calculate the roc metrics
        # fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

        # # Plot the ROC curve
        # plt.plot(fpr,tpr)

        # # Add labels and diagonal line
        # plt.xlabel("False Positive Rate")
        # plt.ylabel("True Positive Rate")
        # plt.plot([0, 1], [0, 1], "k--")
        # plt.show()

        return [accuracy,precision,recall,auc,f1_score_val], y_pred

    def pick_best_model(self,X,y,X_train,y_train,X_test,y_test):

        classifiers = [LogisticRegression(),RandomForestClassifier(),DecisionTreeClassifier(),SVC(),KNeighborsClassifier()]
        classifiers_names = ['LogisticRegression','RandomForestClassifier','DecisionTreeClassifier','SVC','KNN']
        dict_classifiers = dict(zip(classifiers_names, classifiers))
        
        results = {}
        predictions = {}
        best_score = 0
        best_model = ""
        for name, clf in dict_classifiers.items():
            print(f'Predicting using {name}')
            self.hyperparameter_tuning(name,X, y, X_train,y_train)
            scores, y_pred = self.predictive_model(clf, X, y, X_train,y_train, X_test, y_test)
            results[name] = scores
            predictions[name] = y_pred
            preci = scores[1]
            recall = scores[2]
            score = preci * recall
            # print(f'{name} score is {score}')
            if score >= best_score:
                best_score = score
                best_model = name
            # print(f'Current Best model {best_model}')

        model_selected = dict_classifiers[name]
        # print(f'Model Selected: {best_model}')
        return model_selected,best_model,preci,recall

    def predictive_model_final(self,model,X_train,y_train,X_test,y_test):
    
        # Fit to the training data
        model.fit(X_train,y_train)

        # Compute accuracy
        accuracy = model.score(X_test,y_test)

        # Predict the labels of the test set
        y_pred = model.predict(X_test)


        return y_pred

def run_predictor(p1_name,p2_name,matches,players_dict,tourn_info,match_round):
    tc = TennisPredModel(p1_name,p2_name,matches,players_dict,tourn_info,match_round)

    tc.build_data_model()
    tc.prep_features()
    X, y, X_train, X_test, y_train, y_test, X_to_predict, y_to_predict = tc.prep_model()

    model_selected,model_name,preci,recall = tc.pick_best_model(X, y, X_train, y_train, X_test, y_test)
    
    result = tc.predictive_model_final(model_selected,X,y,X_to_predict,y_to_predict)

    if result == 1:
        winner_id = tc.p1_id
        print(f'{tc.p1_id} is going to WIN the match against {tc.p2_id}!')
    else:
        winner_id = tc.p2_id

        print(f'{tc.p1_id} is going to LOSE the match against {tc.p2_id}!')

    return winner_id,model_name,preci,recall

def run_predictor_tournament(p1_name,p2_name,matches,players_dict,tourn_info,rounds):
    data = []
    for t_round in rounds:
       winner_id,model_name,preci,recall = run_predictor(p1_name,p2_name,matches,players_dict,tourn_info,t_round)
       data.append([t_round, winner_id, model_name,round(preci,2),round(recall,2)])
    df = pd.DataFrame(data,columns=['round','winner_id','model','precision','recall'])
    
    return df

# Run app
if __name__=='__main__':
    p1_name = 'Carlos Alcaraz'
    p2_name = 'Cameron Norrie'
    tournament_name = 'Roland Garros'
    tournament_date = date(2023,7,24).strftime("%Y-%m-%d")

    tournament_points = 2000
    tournament_surface = 'Clay'
    infos = [tournament_surface,tournament_points,tournament_date]

    tournament_dict = {}
    tournament_dict[tournament_name] = infos
    tournament_info = tournament_dict[tournament_name]
    tournament_info.append(tournament_name)

    rounds = list(range(0,8))

    matches, rankings, players = data_import()
    players_dict, tourn_dict, rounds_list, matches = get_more_info(matches,rankings,players)
 
    df = run_predictor_tournament(p1_name,p2_name,matches,players_dict,tournament_info,rounds)
    print(df)
