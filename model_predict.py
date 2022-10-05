from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
from datetime import date

from data_prep import data_cleaner

class TennisPredModel():
    def build_data_model(self,df,name_p1,name_p2,rank_p1,rank_p2,tournament_date,tournament_points,tournament_phase,tournament_surface):
        self.data = df
        self.name_p1 = name_p1
        self.name_p2 = name_p2

        self.prepare_match_to_predict(rank_p1,rank_p2,tournament_date,tournament_points,tournament_phase,tournament_surface)
        self.get_matches_from_both()
        self.join_all_matches()

    def prepare_match_to_predict(self,rank_p1,rank_p2,tournament_date,tournament_points,tournament_phase,tournament_surface):
        columns = ['Date','Winner','Loser','WRank','LRank','SeriesPoints','RoundDraw','Type']
        inputs_match = [tournament_date,self.name_p1,self.name_p2,rank_p1,rank_p2,tournament_points,tournament_phase,tournament_surface]
        df_input_match = pd.DataFrame([inputs_match],columns=columns)
        self.match_to_predict = df_input_match

    def get_matches_from_both(self):
        matches_player = self.data[(self.data['H2H'].str.contains(self.name_p1)) | (self.data['H2H'].str.contains(self.name_p2))]
        self.matches_both_players = matches_player

    def join_all_matches(self):
        matches = pd.concat([self.matches_both_players,self.match_to_predict])
        matches = matches.reset_index(drop=True)
        matches = matches.reset_index()
        self.all_matches = matches

    def prep_features(self):
        self.all_matches['Date'] = pd.to_datetime(self.all_matches['Date']).dt.date
        self.all_matches['Old'] = (date.today() - self.all_matches['Date']).dt.days
        type_dummy = pd.get_dummies(self.all_matches['Type'])

        opponents =  pd.DataFrame(np.where((self.all_matches['Winner']==self.name_p1)|(self.all_matches['Winner']==self.name_p2),self.all_matches['Loser'],self.all_matches['Winner']),columns=['Opponent'])
        opponents_dummy = pd.get_dummies(opponents['Opponent'])

        features = ['WRank','LRank','SeriesPoints','RoundDraw']
        scl = StandardScaler()
        matches_scaled = scl.fit_transform(self.all_matches[features])
        matches_scaled_df = pd.DataFrame(matches_scaled, columns=features)

        goal = pd.DataFrame(np.where((self.all_matches['Winner']==self.name_p1)|(self.all_matches['Winner']==self.name_p2),1,0),columns=['Winner'])
        final_df = pd.concat([matches_scaled_df,type_dummy,goal,opponents_dummy,self.all_matches['index']],axis=1)
        self.scaled_data = final_df

    def prep_model(self,train_size_val=0.6,random_state_val=10):
        match_to_predict = pd.DataFrame(self.scaled_data.iloc[-1]).T
        matches_ready = self.scaled_data.iloc[0:-1]

        def create_feature_target_var(df):
            # Create feature variable
            X = df.drop('Winner', axis=1)
            # Create target variable
            y = df['Winner']
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

def run_predictor(name_p1,name_p2,rank_p1,rank_p2,tournament_date,tournament_points,tournament_phase,tournament_surface):
    tennis_clean = data_cleaner() 
    tc = TennisPredModel()
    tc.build_data_model(tennis_clean,name_p1,name_p2,rank_p1,rank_p2,tournament_date,tournament_points,tournament_phase,tournament_surface)
    tc.prep_features()
    X, y, X_train, X_test, y_train, y_test, X_to_predict, y_to_predict = tc.prep_model()

    model_selected,model_name,preci,recall = tc.pick_best_model(X_train, y_train, X_test, y_test)
    
    result = tc.predictive_model_final(model_selected,X,y,X_to_predict,y_to_predict)

    if result == 1:
        winner_name = tc.name_p1
        print(f'{tc.name_p1} is going to WIN the match against {tc.name_p2}!')
    else:
        winner_name = tc.name_p2

        print(f'{tc.name_p1} is going to LOSE the match against {tc.name_p2}!')

    return winner_name,model_name,preci,recall

def run_predictor_tournament(name_p1,name_p2,rank_p1,rank_p2,tournament_date,tournament_points,tournament_surface):
    data = []
    for t_round in range(1,8):
       winner_name,model_name,preci,recall = run_predictor(name_p1,name_p2,rank_p1,rank_p2,tournament_date,tournament_points,t_round,tournament_surface)
       data.append([t_round, winner_name,model_name,preci,recall])
    df = pd.DataFrame(data,columns=['Round','Winner','Model','Precision','Recall'])
    
    return df

# Run app
if __name__=='__main__':
    name_p1,rank_p1 = 'Nadal R.',5
    name_p2,rank_p2 = 'Federer R.',3

    tournament_date = date(2023,7,24)
    tournament_points = 2000
    tournament_phase = 7
    tournament_surface = 'Clay'
    # run_predictor(name_p1,name_p2,rank_p1,rank_p2,tournament_date,tournament_points,tournament_phase,tournament_surface)
    df = run_predictor_tournament(name_p1,name_p2,rank_p1,rank_p2,tournament_date,tournament_points,tournament_surface)
    print(df)
