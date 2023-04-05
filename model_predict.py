from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Import scores
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score


# Import classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
import numpy as np
from datetime import datetime,date

from data_prep import data_import

class TennisPredModel():
    def build_data_model(self,df,p1_id,p2_id,rank_p1,rank_p2,tourn_info,match_round):
        self.data = df
        self.p1_id = p1_id
        self.p2_id = p2_id

        features = ['tourney_date','tourney_name','winner_id','loser_id','winner_rank','loser_rank','tourney_points','round_level','surface']
        self.prepare_match_to_predict(rank_p1,rank_p2,tourn_info,match_round,features)
        self.get_matches_from_both(features)
        self.join_all_matches()
        print(self.all_matches)

    def prepare_match_to_predict(self,rank_p1,rank_p2,tourn_info,match_round,features):
        surface = tourn_info[0]
        # date = tourn_info[2]
        date = datetime.today().strftime("%Y-%m-%d")
        points = tourn_info[1]
        name = tourn_info[3]

        inputs_match = [date, name, self.p1_id, self.p2_id, rank_p1, rank_p2, points, match_round, surface]
        df_input_match = pd.DataFrame([inputs_match],columns=features)
        df_input_match['h2h'] = 1
        self.match_to_predict = df_input_match

    def get_matches_from_both(self,features):
        condition_1 = (self.data['winner_id']== self.p1_id) | (self.data['loser_id']== self.p2_id)
        condition_2 = (self.data['winner_id']== self.p2_id) | (self.data['loser_id']== self.p1_id)
        matches_player = self.data[(condition_1)|(condition_2)]

        matches_player = matches_player[features]
        condition_1 = (matches_player['winner_id']== self.p1_id) & (matches_player['loser_id']== self.p2_id)
        condition_2 = (matches_player['winner_id']== self.p2_id) & (matches_player['loser_id']== self.p1_id)

        matches_player['h2h'] = np.where(((condition_1) | (condition_2)),1,0)

        self.matches_both_players = matches_player

    def join_all_matches(self):
        matches = pd.concat([self.matches_both_players,self.match_to_predict])
        matches = matches.reset_index(drop=True)
        matches = matches.reset_index()
        self.all_matches = matches

    def prep_features(self):
        self.all_matches['old'] = (datetime.now() - pd.to_datetime(self.all_matches['tourney_date'],format='%Y-%m-%d')).dt.days
        print(self.all_matches)
        type_dummy = pd.get_dummies(self.all_matches['surface'])

        encoder = LabelEncoder()
        self.all_matches['tourney_name_enc'] = encoder.fit_transform(self.all_matches['tourney_name'])

        features_to_scale = ['winner_rank','loser_rank','tourney_points','round_level','tourney_name_enc','old']
        scl = StandardScaler()
        matches_scaled = scl.fit_transform(self.all_matches[features_to_scale])
        matches_scaled_df = pd.DataFrame(matches_scaled, columns=features_to_scale)

        goal = pd.DataFrame(np.where((self.all_matches['winner_id']==self.p1_id),1,0),columns=['result'])
        features_not_to_scale = ['index','h2h']
        matches_not_scaled_df = self.all_matches[features_not_to_scale]

        final_df = pd.concat([matches_scaled_df,type_dummy,goal,matches_not_scaled_df],axis=1)
        self.scaled_data = final_df

    def prep_model(self,train_size_val=0.6,random_state_val=10):
        match_to_predict = pd.DataFrame(self.scaled_data.iloc[-1]).T
        matches_ready = self.scaled_data.iloc[0:-1]

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


    def predictive_model(self,model,X_train,y_train,X_test,y_test):
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

    def pick_best_model(self,X_train,y_train,X_test,y_test):
        classifiers = [LogisticRegression(),RandomForestClassifier(),DecisionTreeClassifier()]
        classifiers_names = ['LogisticRegression','RandomForestClassifier','DecisionTreeClassifier']
        dict_classifiers = dict(zip(classifiers_names, classifiers))
        
        results = {}
        predictions = {}
        best_score = 0
        best_model = ""
        for name, clf in dict_classifiers.items():
            scores, y_pred = self.predictive_model(clf,X_train,y_train,X_test,y_test)
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

def run_predictor(matches,p1_id,p2_id,p1_rank,p2_rank,tourn_info,match_round):
    tc = TennisPredModel()
    tc.build_data_model(matches,p1_id,p2_id,p1_rank,p2_rank,tourn_info,match_round)
    tc.prep_features()
    X, y, X_train, X_test, y_train, y_test, X_to_predict, y_to_predict = tc.prep_model()

    model_selected,model_name,preci,recall = tc.pick_best_model(X_train, y_train, X_test, y_test)
    
    result = tc.predictive_model_final(model_selected,X,y,X_to_predict,y_to_predict)

    if result == 1:
        winner_id = tc.p1_id
        print(f'{tc.p1_id} is going to WIN the match against {tc.p2_id}!')
    else:
        winner_id = tc.p2_id

        print(f'{tc.p1_id} is going to LOSE the match against {tc.p2_id}!')

    return winner_id,model_name,preci,recall

def run_predictor_tournament(matches,p1_id,p2_id,p1_rank,p2_rank,tourn_info,rounds):
    data = []
    for t_round in rounds:
       winner_id,model_name,preci,recall = run_predictor(matches,p1_id,p2_id,p1_rank,p2_rank,tourn_info,t_round)
       data.append([t_round, winner_id, model_name,round(preci,2),round(recall,2)])
    df = pd.DataFrame(data,columns=['round','winner_id','model','precision','recall'])
    
    return df

# Run app
if __name__=='__main__':
    p1_id,rank_p1 = 104745,5 # rafa
    # p2_id,rank_p2 = 103819,3 #roger
    p2_id, rank_p2 = 207989,2

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
    matches,ranks,players = data_import() 

    df = run_predictor_tournament(matches,p1_id,p2_id,rank_p1,rank_p2,tournament_info,rounds)
    print(df)
