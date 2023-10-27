import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
#matches_data = pd.read_csv('Dataset/IPL_Matches_2008_2022.csv')
#match_data = pd.read_csv('Dataset/IPL_Ball_by_Ball_2008_2022.csv')

#keeping only required columns helpful in getting target score
class preprocessing():

   def __init__(self,data,file_object,logger_object):
    self.data = data
    self.file_object = file_object
    self.logger_object = logger_object
   def preprocessor(self):
      self.data.drop(columns = ['Season', 'Date','MatchNumber','City','TossWinner','TossDecision','Team1','WinningTeam','WonBy','Margin','method','Player_of_Match','Team1Players','Team2Players','Umpire1','Umpire2'],inplace=True)
      self.data.drop(columns =['non_boundary','extra_type','player_out','kind','fielders_involved'] ,inplace = True)
      agg_data = self.data
      agg_data.BattingTeam = agg_data.BattingTeam.replace(to_replace='Rising Pune Supergiant', value='Rising Pune Supergiants')
      agg_data.BattingTeam = agg_data.BattingTeam.replace(to_replace='Delhi Daredevils', value='Delhi Capitals')
      agg_data.BattingTeam = agg_data.BattingTeam.replace(to_replace='Deccan Chargers', value='Sunrisers Hyderabad')
      agg_data.BattingTeam = agg_data.BattingTeam.replace(to_replace='Kings XI Punjab', value='Punjab Kings')
      
      agg_data.Team2 = agg_data.Team2.replace(to_replace='Rising Pune Supergiant', value='Rising Pune Supergiants')
      agg_data.Team2 = agg_data.Team2.replace(to_replace='Delhi Daredevils', value='Delhi Capitals')
      agg_data.Team2 = agg_data.Team2.replace(to_replace='Deccan Chargers', value='Sunrisers Hyderabad')
      agg_data.Team2 = agg_data.Team2.replace(to_replace='Kings XI Punjab', value='Punjab Kings')
      
      consistent_teams = ['Rajasthan Royals', 
             'Royal Challengers Bangalore', 
             'Sunrisers Hyderabad', 'Punjab Kings', 'Delhi Capitals',
             'Mumbai Indians', 'Chennai Super Kings', 'Kolkata Knight Riders']
      agg_data = agg_data[(agg_data['BattingTeam'].isin(consistent_teams)) & (agg_data['Team2'].isin(consistent_teams))]
      
      normal_match_data = agg_data.loc[agg_data['innings']<3]
      normal_match_data = normal_match_data.reset_index(drop=True)
      normal_match_data['total_wickets'] = normal_match_data.groupby(['ID','BattingTeam'])['isWicketDelivery'].cumsum()
      normal_match_data['total_runs'] = normal_match_data.groupby(['ID','BattingTeam'])['total_run'].cumsum()
      normal_match_data['total_extra_runs'] = normal_match_data.groupby(['ID','BattingTeam'])['extras_run'].cumsum()
      normal_match_data['bt_run']=normal_match_data.groupby(['ID','BattingTeam','batter'])['batsman_run'].cumsum()
      
      normal_match_data['p1'] = normal_match_data['batter'].shift(1)
      normal_match_data['p2'] = normal_match_data['non-striker'].shift(1)
      normal_match_data.loc[0,['p1','p2']] = normal_match_data.loc[0,['batter','non-striker']].values
      normal_match_data['wicket_s']=normal_match_data['total_wickets'].astype(str).shift(1)
      normal_match_data['iswicket_s']=normal_match_data['isWicketDelivery'].astype(str).shift(1)
      normal_match_data['iswicket_s'][0]='0'
      normal_match_data.loc[(normal_match_data['ballnumber']==1)&(normal_match_data['overs']==0),'wicket_s']='0'
      def players(batter,p1,p2,non_striker):
          #print(batter,p1,bt_run)
          p1_n = None
          p2_n = None
          if (batter != p1) and (batter != p2):
              p1_n = batter
              p2_n = non_striker
          elif (non_striker != p2) and (non_striker != p1):
              p1_n = non_striker
              p2_n = batter
              
          return np.array([p1_n,p2_n])
      pl_list=np.vectorize(players,signature='(),(),(),()->(n)')(normal_match_data['batter'],normal_match_data['p1'],normal_match_data['p2'],normal_match_data['non-striker'])
      
      normal_match_data[['p1','p2']]=pl_list
      normal_match_data.loc[0,['p1','p2']] = normal_match_data.loc[0,['batter','non-striker']].values
      #normal_match_data.loc[(normal_match_data['ballnumber']==1)&(normal_match_data['overs']==0),['p1','p2']]=normal_match_data.loc[(normal_match_data['ballnumber']==1)&(normal_match_data['overs']==0),['batter','non-striker']].values
      normal_match_data[['p1','p2']] = normal_match_data.groupby(['ID','BattingTeam','wicket_s'])[['p1','p2']].transform(lambda x: x.ffill())
      
      def bt_run(batter,p1,bt_run):
          #print(batter,p1,bt_run)
        
          if batter==p1:
              bt1_run = bt_run
              bt2_run = None
          else:
              bt2_run = bt_run
              bt1_run = None
          return np.array([bt1_run,bt2_run])
        
      
      run_list=np.vectorize(bt_run,signature='(),(),()->(n)')(normal_match_data['batter'],normal_match_data['p1'],normal_match_data['bt_run'])
      normal_match_data[['bt1_run','bt2_run']]=run_list
      
      normal_match_data.loc[(normal_match_data['overs']==0) & (normal_match_data['ballnumber']==1),['bt1_run','bt2_run']]=0
      normal_match_data.loc[(normal_match_data['iswicket_s']=='1')&(normal_match_data['bt1_run'].isna()),'bt1_run']=0
      #normal_match_data.loc[(normal_match_data['newplayer']==normal_match_data['p2'])&(normal_match_data['bt2_run'].isna()),'bt2_run']=0
      normal_match_data[['bt1_run']]=normal_match_data.groupby(['ID','BattingTeam','p1'])[['bt1_run']].transform(lambda x: x.ffill())
      normal_match_data[['bt1_run']]=normal_match_data.groupby(['ID','BattingTeam','p1'])[['bt1_run']].transform(lambda x: x.bfill())
      
      normal_match_data[['bt2_run']]=normal_match_data.groupby(['ID','BattingTeam','p2'])[['bt2_run']].transform(lambda x: x.ffill())
      normal_match_data.loc[(normal_match_data['p1'].shift(1)==normal_match_data['p2'])&(normal_match_data['bt2_run'].isna()),'bt2_run']=normal_match_data.loc[((normal_match_data['p1'].shift(1)==normal_match_data['p2']).shift(-1))&(normal_match_data['bt2_run'].shift(-1).isna()),'bt1_run'].values
      normal_match_data[['bt2_run']]=normal_match_data.groupby(['ID','BattingTeam','p2'])[['bt2_run']].transform(lambda x: x.ffill())
      
      normal_match_data.drop(columns=['batsman_run', 'extras_run','isWicketDelivery','bt_run', 'p1', 'p2',
             'wicket_s', 'iswicket_s','SuperOver'],inplace=True)
      
      normal_match_data['overs'] = normal_match_data['overs']+normal_match_data['ballnumber']*0.1
      
      normal_match_data.drop(columns=['ballnumber','total_run'],inplace=True)
      normal_match_data['score'] =  normal_match_data.groupby(['ID','innings'])['total_runs'].transform('max')
      self.scaler = StandardScaler()
      num_df = normal_match_data[['overs', 'total_wickets', 'total_runs','total_extra_runs', 'bt1_run', 'bt2_run']]
      self.scaled_data = self.scaler.fit_transform(num_df)
      #self.scaled_num_df = pd.DataFrame(data=self.scaled_data, columns=num_df.columns)
      normal_match_data[['overs', 'total_wickets', 'total_runs','total_extra_runs', 'bt1_run', 'bt2_run']]= self.scaled_data
      normal_match_data.drop(columns=['ID','innings','batter','bowler','non-striker','Venue','total_runs'],inplace=True)
      self.encoded_df = pd.get_dummies(data=normal_match_data, columns=['BattingTeam', 'Team2'])
      return self.encoded_df

      
   def separate_label_feature(self, data, label_column_name):
       """
                       Method Name: separate_label_feature
                       Description: This method separates the features and a Label Coulmns.
                       Output: Returns two separate Dataframes, one containing features and the other containing Labels .
                       On Failure: Raise Exception   
                       Written By: iNeuron Intelligence
                       Version: 1.0
                       Revisions: None   
               """
       self.logger_object.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
       try:
           self.X=data.drop(labels=label_column_name,axis=1) # drop the columns specified and separate the feature columns
           self.Y=data[label_column_name] # Filter the Label columns
           self.logger_object.log(self.file_object,
                                  'Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class')
           return self.X,self.Y
       except Exception as e:
           self.logger_object.log(self.file_object,'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(e))
           self.logger_object.log(self.file_object, 'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
           raise Exception()
   
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from file_operations import file_methods

class KMeansClustering:
    """
            This class shall  be used to divide the data into clusters before training.

            Written By: iNeuron Intelligence
            Version: 1.0
            Revisions: None

            """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def elbow_plot(self,data):
        """
                        Method Name: elbow_plot
                        Description: This method saves the plot to decide the optimum number of clusters to the file.
                        Output: A picture saved to the directory
                        On Failure: Raise Exception

                        Written By: iNeuron Intelligence
                        Version: 1.0
                        Revisions: None

                """
        self.logger_object.log(self.file_object, 'Entered the elbow_plot method of the KMeansClustering class')
        wcss=[] # initializing an empty list
        try:
            for i in range (1,11):
                kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42) # initializing the KMeans object
                kmeans.fit(data) # fitting the data to the KMeans Algorithm
                wcss.append(kmeans.inertia_)
            plt.plot(range(1,11),wcss) # creating the graph between WCSS and the number of clusters
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            #plt.show()
            plt.savefig('K-Means_Elbow.PNG') # saving the elbow plot locally
            # finding the value of the optimum cluster programmatically
            self.kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
            self.logger_object.log(self.file_object, 'The optimum number of clusters is: '+str(self.kn.knee)+' . Exited the elbow_plot method of the KMeansClustering class')
            return self.kn.knee

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in elbow_plot method of the KMeansClustering class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Finding the number of clusters failed. Exited the elbow_plot method of the KMeansClustering class')
            raise Exception()

    def create_clusters(self,data,number_of_clusters):
        """
                                Method Name: create_clusters
                                Description: Create a new dataframe consisting of the cluster information.
                                Output: A datframe with cluster column
                                On Failure: Raise Exception

                                Written By: iNeuron Intelligence
                                Version: 1.0
                                Revisions: None

                        """
        self.logger_object.log(self.file_object, 'Entered the create_clusters method of the KMeansClustering class')
        self.data=data
        try:
            self.kmeans = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=42)
            #self.data = self.data[~self.data.isin([np.nan, np.inf, -np.inf]).any(1)]
            self.y_kmeans=self.kmeans.fit_predict(data) #  divide data into clusters

            self.file_op = file_methods.File_Operation(self.file_object,self.logger_object)
            self.save_model = self.file_op.save_model(self.kmeans, 'KMeans') # saving the KMeans model to directory
                                                                                    # passing 'Model' as the functions need three parameters

            self.data['Cluster']=self.y_kmeans  # create a new column in dataset for storing the cluster information
            self.logger_object.log(self.file_object, 'succesfully created '+str(self.kn.knee)+ 'clusters. Exited the create_clusters method of the KMeansClustering class')
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in create_clusters method of the KMeansClustering class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Fitting the data to clusters failed. Exited the create_clusters method of the KMeansClustering class')
            raise Exception()