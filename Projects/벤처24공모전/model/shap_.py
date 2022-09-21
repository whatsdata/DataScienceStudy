import shap 
import numpy as np
import pandas as pd

def make_df(value, X):
    return pd.DataFrame(np.mean(np.abs(value),axis=0), index = X.columns)

class shap_():
    def __init__(self, model,  X,y, X_export,y_export):
        '''
        model : 사용할 모델 (e.g. catboost, xgboost... training 된 모델 투입)
        X : 사용할 독립변수 (비수출 기업)
        y : 사용할 종속변수
        X_export : 수출기업에 대한 정보(X)
        y_export : 수출기업에 대한 정보(y)
        '''
        self.model = model
        self.X = X
        self.y = y
        self.X_export = X_export
        self.y_export = y_export
        
        # Generate the Tree SHAP estimator of Shapley values that corresponds to the Random Forest we built
        self.explainer = shap.TreeExplainer(model, feature_perturbation='interventional')
        # Compute the estimated Shapley values for the test sample's observations
        self.shap_values = self.explainer.shap_values(X)
        
        self.sectors = ['energy/medical','computer','telecommunication', 'food/clothes','manufature', 'software','IT', 'Other' ]
        self.meanscore = self.score_by_sector()
        
    def waterfall(self, index, max_display = 30 ):
        shap.plots._waterfall.waterfall_legacy(self.explainer.expected_value, self.shap_values[index],self.X.iloc[index, :],max_display = max_display)
        
    def summary_plot(self ,max_display = 30):
        shap.summary_plot(self.shap_values,self.X, max_display = max_display, plot_type = 'bar')
        
        
    def shap_by(self):
        shaps = make_df(self.shap_values, shap_X)
                        
        characteristic = shaps.iloc[0]+shaps.iloc[1]+shaps.iloc[3]+shaps.iloc[4]
        manager = shaps.iloc[5]+shaps.iloc[6]+shaps.iloc[7]+shaps.iloc[8]
        financial = shaps.iloc[9]+shaps.iloc[10]+shaps.iloc[11]+shaps.iloc[12]+shaps.iloc[13]+shaps.iloc[14]
        technical = shaps.iloc[15]+shaps.iloc[16]+shaps.iloc[17]+shaps.iloc[18]+shaps.iloc[19]+shaps.iloc[2]
        marketing = shaps.iloc[20]+shaps.iloc[21]+shaps.iloc[22]+shaps.iloc[23]

        return pd.DataFrame([characteristic[0],financial[0],manager[0],marketing[0],technical[0]], index = ['characteristic','financial','manager','marketing','technical'])
    
        
    def expectedscore(self, index):
        return np.round(self.explainer.expected_value + np.sum(self.shap_values[index]),2)
    
    # def score_by_sector(self):
    #     list = []
    #     for i in np.unique(self.X_export['sectors']):
    #         list.append(np.quantile(self.y_export[(self.X_export['profit']>0) & (self.X_export['sectors'] == i)], 0.5))
    #     meanscore = pd.DataFrame(list, index = ['에너지/의료/정밀','컴퓨터/반도체/전자부품', '통신기기/방송기기', '음식료/섬유/(비)금속',' 기계/제조/자동차' ,'소프트웨어개발','정보통신/방송서비스', '기타'])
    #     return meanscore
    
    def score_by_sector(self):
        list = []
        for i in self.sectors:
          list.append(np.quantile(self.y_export[(self.X_export['gainincrease']>1) & (self.X_export['sectors'] == i)], 0.5))
        meanscore = pd.DataFrame(list, index = ['에너지/의료/정밀','컴퓨터/반도체/전자부품', '통신기기/방송기기', '음식료/섬유/(비)금속',' 기계/제조/자동차' ,'소프트웨어개발','정보통신/방송서비스', '기타'])
        return meanscore

    def rate(self,index):
        
        expectedscore = self.expectedscore(index)
        print('-------',index,'-------')
        
        print('수출지수 임계점 : ',self.meanscore.iloc[self.sectors.index(self.X.iloc[index]['sectors'])][0] )
        print('수출지수        : ', expectedscore)
        if self.meanscore.iloc[self.sectors.index(self.X.iloc[index]['sectors'])][0]<expectedscore:
            print('pass? : Yes')
        else:
            print('pass? : No')  
        
        

