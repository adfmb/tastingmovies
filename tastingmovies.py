import pandas as pd
from IPython.display import HTML
import nltk 
import numpy as np
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
import pyspark
sc = pyspark.SparkContext
from pyspark.sql import SparkSession
from pyspark import SQLContext
from pyspark.sql import Window as W
import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import *
from pyspark.sql import *



class prepmovies():

    tag_init = None
    minimun_wordfreq = None
    stemmer = None
    stop_words = None
    rating = None
    genome_tags_relevance = None
    movie = None
    minimun_to_hr=None
    key_columns = None
    spark = None

    def __init__(self,tag_init,minimun_wordfreq,rating,genome_tags_relevance,#genome_scores,genome_tags,
        movie,spark,
        minimun_to_hr = 4,stop_words=None,stemmer=None,key_columns = ['userId', 'movieId','title', 'timestamp_TAG', 'timestamp_rating','timestamp_movie', 'rating_usr','high_rating']):

        self.spark = spark
        self.tag_init = tag_init
        self.minimun_wordfreq = minimun_wordfreq
        self.minimun_to_hr = minimun_to_hr
        self.key_columns = key_columns

        if rating is not None:
            self.rating = rating \
                .rename(columns={'timestamp':'timestamp_rating', 'rating':'rating_usr'})

        self.genome_tags_relevance = genome_tags_relevance

        #movie['I']=1
        if movie is not None:
            movie['genres_list'] = movie.genres.str.split('|',expand=False)

        self.movie = movie


        if(stop_words is None):
            nltk.download('wordnet')
            self.stop_words = set(stopwords.words("english")) 

        if stemmer  is None :
            self.stemmer = WordNetLemmatizer()

        pass



    def join_tags_and_genome_ratings(self):

        print('join_tags_and_genome_ratings')
        #genome_tags_relevance=pd.merge(self.genome_tags,self.genome_scores,how='left',on=['tagId'])

        tag_relevance = pd.merge(self.tag_init,self.genome_tags_relevance,how='left',on=['tag','movieId'])

        #tag_relevance['relevance'] = tag_relevance.relevance.fillna(.01)


        tags_rating00 = pd.merge(tag_relevance, self.rating, how="left", on=['userId','movieId']).sort_values(['timestamp','userId','movieId'],ascending=[1,0,0])

        tags_rating = tags_rating00[tags_rating00.rating_usr.notnull()]

        pre_tag=pd.merge(tags_rating,self.movie.drop(columns=['genres']),how='left',on=['movieId'])


        '''
        Rellenar los NA con el valor de 1% bajo la lógica que si no tiene un valor de relevancia asociado es 
        porque no es tan relevante para esa película y, para no matar el valor por el que luego quisiéramos 
        multiplicar a esa relevancia, no la volvemos 0 sino que la dejamos en un valor bajo como puede ser 1%
        '''
        

        pre_tag['tag_relevance_movie']  = pre_tag['relevance'].fillna(.01)#*pre_tag['rating_usr'] 

        min_timestamps = pre_tag.groupby(['userId','movieId']).agg(
                    {
                        'timestamp':'min',
                        'timestamp_rating':'min'
                    }
                ).reset_index()

        min_timestamps['timestamp_movie'] = min_timestamps[['timestamp','timestamp_rating']].min(axis=1)



        self.tag = pd.merge(min_timestamps[['userId','movieId','timestamp_movie']],pre_tag,on=['userId','movieId'],how='left')

        return(self.tag)

    def preproc_text(self,text_tmp): #,documents)  


        #print('Remove all the special characters')
        document = re.sub(r'\W', ' ', str(text_tmp))
        
        #print('Remove all single characters')
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        
        #print('Remove single characters from the start')
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
        
        #print('Substituting multiple spaces with single space')
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        
        #print('Removing prefixed "b" ')
        document = re.sub(r'^b\s+', '', document)
        
        #print('Converting to Lowercase')
        document = document.lower()
        
        #print('Lemmatization')
        document = document.split()
        
        document = [self.stemmer.lemmatize(word) for word in document]
        document = [word for word in document if not word in self.stop_words]
        document = ' '.join(document)
        
        #documents.append(document)
        return(document)


    def process_tags01(self):
        
        print('---Remove all the special characters')
        print('------Remove all single characters')
        print('---------Remove single characters from the start')
        print('------------Substituting multiple spaces with single space')
        print('---------------Removing prefixed "b" ')
        print('---------------------Converting to Lowercase')
        print('------------------------Lemmatization')

        self.tag['tag_clean'] = [self.preproc_text(x) for x in self.tag.tag]
        self.tag['tags_list'] = self.tag.tag_clean.str.split(expand=False)

        counts_tags = self.tag.tag_clean.str.split(expand=True).stack().value_counts()
        df_counts_tags = counts_tags.to_frame()
        df_counts_tags.columns = ['frec_tag']
        df_counts_tags['word'] = df_counts_tags.index
        df_counts_tags.reset_index(drop=True, inplace=True)
        df_counts_tags['num_charac'] = [len(x) for x in df_counts_tags.word]

        df_counts_tags01 = df_counts_tags[(df_counts_tags.num_charac>1) & (df_counts_tags.frec_tag>self.minimun_wordfreq)]

        def filter_tags(tagtmp):
            return([x for x in df_counts_tags01.word if x in tagtmp])

        self.tag['tags_2consider'] = [filter_tags(a) for a in self.tag.tags_list]
        self.tag['num_words'] = [len(x) for x in self.tag.tags_2consider]
        #self.tag['timestamp_movie'] = self.tag[['timestamp','timestamp_rating']].min(axis=1)

        tag_sub = self.tag[self.tag.num_words>0][['userId','movieId','timestamp_movie','tags_2consider','rating_usr','relevance']]
        tag_sub['I'] = 1

        tag_sub_movies = self.tag[self.tag.num_words>0][['userId','movieId','title','timestamp_movie','rating_usr','genres_list']]
        tag_sub_movies['I'] = 1

        self.tag_sub = tag_sub
        self.tag_sub_movies=tag_sub_movies



    def pivoting_tag_relevance_by_movie(self):

        print('Calculating: catalogue/ dictionary by pivoting table with tags + relevance by movie, not userId hence, the relevance is static and does not change with the users neither with time')

        tag_relevance_stacked = self.tag_sub.drop(columns=['userId','I','rating_usr','timestamp_movie']).set_index(['movieId','relevance']) \
                                .tags_2consider \
                                .apply(pd.Series) \
                                .stack() \
                                .reset_index() \
                                .rename(columns={0:'tags_2consider'}) 

        tag_relevance_pivot = pd.pivot_table(tag_relevance_stacked,index=['movieId'],columns='tags_2consider',
                                                            values=['relevance'],
                                                            aggfunc={
                                                                    #'I' : np.max,
                                                                    'relevance' : np.mean # the mean will function in thise case and it is logic since will make an averange of the relevence of a word and its differents tags
                                                                    },
                                                            fill_value=0
                                                            ).reset_index() 
       # \.rename(columns={'timestamp':'timestamp_TAG'})

        def reindexing_rlvnc(xtmp):
            if xtmp[1] == '':
                res = xtmp[0]
#            elif ((xtmp[1] != '') & (xtmp[0] == 'I')):
#                res = xtmp[1]
            else:
                res = xtmp[1] + '_rlvnc'
            return(res)
        #df.columns = ['_'.join(col) for col in df.columns]
        tag_relevance_pivot.columns = [reindexing_rlvnc(a) for a in tag_relevance_pivot.columns]

        self.tag_relevance_pivot = tag_relevance_pivot

        return(self.tag_relevance_pivot)



    def pivoting_genre_by_movie(self):

        print('Calculating: catalogue/ dictionary by pivoting table with genres by movie, since not userId, the presence of a genre is static and does not change with the users neither with time')

        genre_by_movie_stacked = self.tag_sub_movies.drop(columns=['userId','rating_usr','timestamp_movie','title']).set_index(['movieId','I']) \
                                .genres_list \
                                .apply(pd.Series) \
                                .stack() \
                                .reset_index() \
                                .rename(columns={0:'genres_list'}) 

        genre_by_movie_pivot = pd.pivot_table(genre_by_movie_stacked,index=['movieId'],columns='genres_list',
                                                            values=['I'],
                                                            aggfunc={
                                                                    'I' : np.max,
                                                                    #'relevance' : np.mean # the mean will function in thise case and it is logic since will make an averange of the relevence of a word and its differents tags
                                                                    },
                                                            fill_value=0
                                                            ).reset_index() 
       # \.rename(columns={'timestamp':'timestamp_TAG'})

        def reindexing_genre(xtmp):
            if xtmp[1] == '':
                res = xtmp[0]
#            elif ((xtmp[1] != '') & (xtmp[0] == 'I')):
#                res = xtmp[1]
            else:
                res = 'genre_' + xtmp[1]
            return(res)
        #df.columns = ['_'.join(col) for col in df.columns]
        genre_by_movie_pivot.columns = [reindexing_genre(a) for a in genre_by_movie_pivot.columns]

        self.genre_by_movie_pivot = genre_by_movie_pivot

        return(self.genre_by_movie_pivot)



    def pivoting_tag_sub(self):

        print('pivoting table with tags + rating')

        tag_sub_stacked = self.tag_sub.drop(columns=['relevance']).set_index(['I','userId','movieId','timestamp_movie','rating_usr']) \
                                .tags_2consider \
                                .apply(pd.Series) \
                                .stack() \
                                .reset_index() \
                                .rename(columns={0:'tags_2consider'}) 

        tag_sub_pivot = pd.pivot_table(tag_sub_stacked,index=['userId','movieId','timestamp_movie'],columns='tags_2consider',
                                                            values=['I','rating_usr'],
                                                            aggfunc={
                                                                    'I' : np.max,
                                                                    'rating_usr' : np.mean
                                                                    },
                                                            fill_value=0
                                                            ).reset_index() 
       # \.rename(columns={'timestamp':'timestamp_TAG'})

#        def reindexing(xtmp):
#            if xtmp[1] == '':
#                res = xtmp[0]
#            elif ((xtmp[1] != '') & (xtmp[0] == 'I')):
#                res = xtmp[1]
#            else:
#                res = xtmp[1] + '_rtng_x_rlvnc'
#            return(res)
#        #df.columns = ['_'.join(col) for col in df.columns]
        tag_sub_pivot.columns = [self.reindexing(a) for a in tag_sub_pivot.columns]

        self.tag_sub_pivot = tag_sub_pivot

        return(self.tag_sub_pivot)



    def pivoting_movies_genres(self):

        print('pivoting ratings + movies´s genres')

        movie_stacked = self.tag_sub_movies.set_index(['I','userId','movieId','title','timestamp_movie','rating_usr']) \
                                .genres_list \
                                .apply(pd.Series) \
                                .stack() \
                                .reset_index() \
                                .rename(columns={0:'genres_list'}) 

        movie_pivot01 = pd.pivot_table(movie_stacked,index=['userId','movieId','timestamp_movie','title'],columns='genres_list',
                                                            values=['I','rating_usr'],
                                                            aggfunc={
                                                                    'I' : np.max,
                                                                    'rating_usr' : np.mean
                                                                    },
                                                            fill_value=0
                                                            ).reset_index()

        #df.columns = ['_'.join(col) for col in df.columns]
        movie_pivot01.columns = [self.reindexing(a) for a in movie_pivot01.columns]

        self.movie_pivot01 = movie_pivot01

        return(self.movie_pivot01)

    def reindexing(self,xtmp):
        if xtmp[1] == '':
            res = xtmp[0]
        elif ((xtmp[1] != '') & (xtmp[0] == 'I')):
            res = xtmp[1]
        else:
            res = xtmp[1] + '_rtng'
        return(res)



    def cumsum_tags_by_user(self):


        self.pivoting_tag_sub()

        key_columns = ['userId', 'movieId','timestamp_movie']
        cols_rtng = [name for name in self.tag_sub_pivot.columns if '_rtng' in name]
        cols_nortng = [re.sub(r'_rtng', '', x) for x in cols_rtng]

        subset = self.tag_sub_pivot[key_columns+cols_rtng+cols_nortng].sort_values(by=['userId','timestamp_movie'])
        subset['I_for_id'] = 1
        gb_subset = subset.groupby(['userId'])
        subset['id_movie_user'] = gb_subset['I_for_id'].cumsum(axis=0)
        df_cumsum_tags_by_user= subset.drop(columns=['movieId','timestamp_movie']).groupby('userId').expanding().sum().drop(columns=['userId','id_movie_user']).rename(columns={'I_for_id':'id_movie_user'}).reset_index().drop(columns=['level_1'])
        self.df_cumsum_tags_by_user = pd.merge(subset[['userId','id_movie_user','movieId']],df_cumsum_tags_by_user,on=['userId','id_movie_user'],how='left')
        #self.tags_rtng=cols_rtng
        #self.tags_nortng=cols_nortng

    def cumsum_genres_by_user(self):


        self.pivoting_movies_genres()

        key_columns = ['userId', 'movieId','timestamp_movie']
        cols_rtng = [name for name in self.movie_pivot01.columns if '_rtng' in name]
        cols_nortng = [re.sub(r'_rtng', '', x) for x in cols_rtng]

        subset = self.movie_pivot01[key_columns+cols_rtng+cols_nortng].sort_values(by=['userId','timestamp_movie'])
        subset['I_for_id'] = 1
        gb_subset = subset.groupby(['userId'])
        subset['id_movie_user'] = gb_subset['I_for_id'].cumsum(axis=0)
        df_cumsum_genres_by_user= subset.drop(columns=['movieId','timestamp_movie']).groupby('userId').expanding().sum().drop(columns=['userId','id_movie_user']).rename(columns={'I_for_id':'id_movie_user'}).reset_index().drop(columns=['level_1'])
        self.df_cumsum_genres_by_user = pd.merge(subset[['userId','id_movie_user','movieId']],df_cumsum_genres_by_user,on=['userId','id_movie_user'],how='left')
        #self.genres_rtng=cols_rtng
        #self.genres_nortng=cols_nortng

    def cumulative_ratings_x_user(self):
        
        spark = self.spark
        self.df_cumsum_tags_by_user_sp = spark.createDataFrame(self.df_cumsum_tags_by_user)

        self.df_cumsum_genres_by_user_sp = spark.createDataFrame(self.df_cumsum_genres_by_user)

        cols_tags=self.df_cumsum_tags_by_user_sp.columns
        cols_genres=self.df_cumsum_genres_by_user_sp.columns
        cols_genres_in_tags=[x for x in cols_genres if x  in cols_tags or x.lower() in cols_tags]

                
        self.df_cumsum_genres_by_user_sp = self.df_cumsum_genres_by_user_sp \
                    .select('*',*[F.col(a).alias(a+ '_genres') for a in cols_genres_in_tags]) \
                    .drop(*cols_genres_in_tags)


        df_cumsum_tags_genres_by_user_sp = self.df_cumsum_tags_by_user_sp \
                                    .join(
                                            self.df_cumsum_genres_by_user_sp,
                                            (
                                                (self.df_cumsum_tags_by_user_sp.userId==self.df_cumsum_genres_by_user_sp.userId_genres) 
                                                & (self.df_cumsum_tags_by_user_sp.id_movie_user==self.df_cumsum_genres_by_user_sp.id_movie_user_genres)
                                            )) \
                                    .drop(*['userId_genres','id_movie_user_genres','movieId_genres'])


        #self.cols_rtng = [name for name in df_cumsum_tags_genres_by_user_sp.columns if '_rtng' in name]
        #self.cols_nortng = [re.sub(r'_rtng', '', x) for x in self.cols_rtng]

        self.cols_rtng = [x for x in df_cumsum_tags_genres_by_user_sp.columns if x.endswith('_rtng')]
        self.cols_rtng_genres = [x for x in df_cumsum_tags_genres_by_user_sp.columns if x.endswith('_rtng_genres')]

        self.cols_nortng = [re.sub(r'_rtng', '', x) for x in self.cols_rtng]
        self.cols_nortng_genres = [re.sub(r'_rtng_genres', '', x) for x in self.cols_rtng_genres]



        #list_rtngacum = [F.round(F.col(x+'_rtng')/F.col(x),2).alias(x+'_avg_rtng_acum') for x in self.cols_nortng]
        list_rtngacum = [F.round(F.col(x+'_rtng')/F.col(x),2).alias(x+'_avg_rtng_acum') for x in self.cols_nortng] + [F.round(F.col(x+'_rtng_genres')/F.col(x+'_genres'),2).alias(x+'_avg_rtng_acum') for x in self.cols_nortng_genres]

        self.df_avgrtng_acum_by_user_pd = df_cumsum_tags_genres_by_user_sp \
                    .select('userId','id_movie_user','movieId',*list_rtngacum) \
                    .toPandas()

    def cumulative_ratings_x_movie_and_rlvnc_genre(self):
        
        print('Calculating the cumulative rating per movie through chronology-ratings')
        subset = self.tag_sub_movies[['userId','movieId','timestamp_movie','rating_usr']].drop_duplicates(subset=['userId','movieId','timestamp_movie']).sort_values(by=['movieId','timestamp_movie'])
        subset['I_for_id'] = 1
        gb_subset = subset.groupby(['movieId'])
        subset['id_user_movie'] = gb_subset['I_for_id'].cumsum(axis=0)
        self.df_cumsum_rating_by_movie= subset.drop(columns=['userId','timestamp_movie']).groupby('movieId').expanding().sum().drop(columns=['movieId','id_user_movie']).rename(columns={'I_for_id':'id_user_movie'}).reset_index().drop(columns=['level_1'])
        self.df_cumsum_rating_by_movie['rating_avg_acum_x_movie'] = self.df_cumsum_rating_by_movie['rating_usr'] / self.df_cumsum_rating_by_movie['id_user_movie']

        self.df_avgrtng_acum_by_movie = pd.merge(subset.drop(columns=['I_for_id']),self.df_cumsum_rating_by_movie.rename(columns={'rating_usr':'cumsum_rating_usrs'}),on=['movieId','id_user_movie'],how='left')


        self.pivoting_tag_relevance_by_movie() #self.tag_relevance_pivot
        self.pivoting_genre_by_movie() #self.genre_by_movie_pivot
        self.df_avgrtng_acum_by_movie_and_rlvnc = pd.merge(self.df_avgrtng_acum_by_movie,self.tag_relevance_pivot,on=['movieId'],how='left')
        self.df_avgrtng_acum_by_movie_and_rlvnc_Igenre = pd.merge(self.df_avgrtng_acum_by_movie_and_rlvnc,self.genre_by_movie_pivot,on=['movieId'],how='left')


    def dataset_for_model(self):

        vars_I_genre_t1 = [x for x in self.df_avgrtng_acum_by_movie_and_rlvnc_Igenre.columns if x.startswith('genre_')]
        vars_tag_relevance_t1 = [x for x in self.df_avgrtng_acum_by_movie_and_rlvnc_Igenre.columns if x.endswith('_rlvnc')]
        pre_tbl_movie_t1 = self.df_avgrtng_acum_by_movie_and_rlvnc_Igenre[
                            ['movieId','id_user_movie','userId','timestamp_movie','rating_usr','rating_avg_acum_x_movie'] + vars_I_genre_t1 + vars_tag_relevance_t1
                            ]
        pre_tbl_movie_t1['id_user_movie_prev'] = pre_tbl_movie_t1['id_user_movie'] -1

        pre_tbl_movie_t1 = pd.merge(
                            pre_tbl_movie_t1,
                            self.df_avgrtng_acum_by_movie_and_rlvnc_Igenre[['movieId','timestamp_movie','id_user_movie','userId','rating_avg_acum_x_movie']] \
                            .rename(columns = {
                                'id_user_movie':'id_user_movie_prev',
                                'rating_avg_acum_x_movie':'rating_avg_acum_x_movie_prev',
                                'userId':'userId_prev',
                                'timestamp_movie':'timestamp_movie_prev'
                            }),
                            how='left', on=['movieId','id_user_movie_prev']
                        )

        tbl_movie_t1 = pre_tbl_movie_t1.drop(columns=['rating_avg_acum_x_movie'])[pre_tbl_movie_t1.id_user_movie_prev>0]

        pre_tbl_usr_t0 = self.df_avgrtng_acum_by_user_pd
        pre_tbl_usr_t0.columns = [ x+'_prev' if x not in ['movieId'] else x for x in pre_tbl_usr_t0.columns]
        
        self.tbl_usrmovie_t0 = pd.merge(tbl_movie_t1,pre_tbl_usr_t0,on=['movieId','userId_prev'],how='inner')
        self.tbl_usrmovie_t0['high_rating'] = [1 if x >=self.minimun_to_hr else 0 for x in self.tbl_usrmovie_t0['rating_usr']]

        self.ds_tastingmovies = self.tbl_usrmovie_t0.drop(columns=['rating_usr','movieId','userId','id_user_movie','id_user_movie_prev','id_movie_user_prev','userId_prev','timestamp_movie','timestamp_movie_prev'])


#        self.df_varsuser_varsmovie = pd.merge(self.df_avgrtng_acum_by_movie_and_rlvnc,self.df_avgrtng_acum_by_user_pd,on=['userId','movieId'],how='inner')
#        #self.df_varsuser_varsmovie = pd.merge(self.df_avgrtng_acum_by_user_pd,self.df_avgrtng_acum_by_movie_and_rlvnc,on=['userId','movieId'],how='inner')

#        vars_usr = [x for x in self.df_varsuser_varsmovie.columns if x.endswith('_avg_rtng_acum')]
#        vars_movie = [x for x in self.df_varsuser_varsmovie.columns if x.endswith('_rlvnc')]
#        vars_faltantes = [x for x in self.df_varsuser_varsmovie.columns if x not in vars_usr + vars_movie]







#    def pivoting_and_BDt0(self):

        

        #spark = self.spark



        #print('merging pivots into BD')
        #BDt0 = pd.merge(self.tag_sub_pivot,self.movie_pivot01, on=['userId','movieId'],how='left')
        #BDt0['timestamp_movie'] = BDt0[['timestamp_TAG','timestamp_rating']].min(axis=1)
        #BDt0['high_rating'] = [1 if x >=self.minimun_to_hr else 0 for x in BDt0['rating_usr']]
        #notkey_columns = [x for x in BDt0.columns if x not in self.key_columns]
        #notkey_columns
 #       self.BDt0 = BDt0[self.key_columns+notkey_columns]


        
#        return(self.BDt0)





#        tag_sub_stacked01 = self.tag_sub.set_index(['I','userId','movieId','timestamp','rating']) \
#                        .tags_2consider \
#                        .apply(pd.Series) \
#                        .stack() \
#                        .reset_index(level=0) \
#                        .rename(columns={0:'tags_2consider'}) 

#        tag_sub_pivot01 = pd.pivot_table(tag_sub_stacked01,index=['userId','movieId','timestamp','rating'],columns='tags_2consider',values=['I'],aggfunc=np.sum).fillna(0).reset_index() \
#                        .rename(columns={'timestamp':'timestamp_TAG'})

