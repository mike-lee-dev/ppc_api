import array
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import re
import math
from filterpy.kalman import KalmanFilter
from filterpy.common import Saver, Q_discrete_white_noise
from ..utils import input_output
from ..helpers import date_helpers

#Todo we should estimate order rate (DT), purchase per order (DT) and value per perchase (price avg) separately
#get parameter out the code:https://www.youtube.com/watch?v=UU7cb1qiS64

def initiate(df_clustered, path):
	K_t=pd.DataFrame(index=df_clustered['Leave'].unique(), columns=['K'])
	P_t=pd.DataFrame(index=df_clustered['Leave'].unique(), columns=['P_t','P_t_prior','P_t_post'])
	Q_t=pd.DataFrame(index=df_clustered['Leave'].unique(), columns=['Q'])
	R_t=pd.DataFrame(index=df_clustered['Leave'].unique(), columns=['R'])
	S_t=pd.DataFrame(index=df_clustered['Leave'].unique(), columns=['S'])
	y_t=pd.DataFrame(index=df_clustered['Leave'].unique(), columns=['y'])
	threshold=50
	#if can find file, read, else create new filter
	try: #still need to be implemented, but we need to save clustering rules first
		R, Q, P, H, F, x_t=input_output.read_kalman_state(path)
		kf, n=create_kalman_filter(x_t)
		kf.R=R
		kf.Q=Q
		kf.P=P
		kf.H=H
		kf.F=F
		kf.x=x_t.to_numpy()
		print(f"Kalman filter could be loaded successfully :{kf}")
		sampled_data=auto_sampling(df_clustered, threshold, i, end_date)
		m_t=get_kalman_measurment(x_t,sampled_data.set_index('Leave'))
		kf=initiate_measurment_covariance(kf, m_t, x_t)
		kf=update(kf, m_t)
		input_output.append_state_history(dates[i-1].date(),x_t, m_t, path)
		print(f"state: {kf.x}")
		kf=predict_and_save(kf, x_t, path)
	except (FileNotFoundError, OSError):	
		print(f"Kalman filter not found, new Kalman filter needs to be created")
		x_t=initiate_kalman_state(df_clustered)
		m_t=get_kalman_measurment(x_t,df_clustered)
		kf, n =create_kalman_filter(x_t)
		kf.x=x_t.to_numpy()
		kf=measurement_function(kf, n)
		kf=initiate_process_matrix(x_t, kf, n)
		kf=initiate_measurment_covariance(kf, m_t, x_t)
		kf=initiate_process_covariance(kf, df_clustered)
		kf=initiate_naive_model(kf,n)
		dic = {}
		look_back_window=14
		
		#dates = pd.date_range(df_clustered.index[14], df_clustered.index[-1])
		dates = df_clustered.sort_index().index.unique()
		#kf, x_t=predict_and_save(kf, x_t)
		for i, end_date in enumerate(dates):#range(look_back_window, len(dates)):
			# df.loc[dates[i]] = [None, None, None]		    
			#sampled_data=constant_sampling(df_clustered, look_back_window, i, dates)
			kf, x_t=predict_and_save(kf, x_t, path)
			sampled_data=auto_sampling(df_clustered, threshold, i, end_date)#pd.to_datetime('2022-03-01', format='%Y-%m-%d'))#end_date)
			m_t=get_kalman_measurment(x_t,sampled_data.set_index('Leave'))
			kf=update_and_save(kf, m_t, x_t)
			K_t[['K']]=np.reshape(kf.K.diagonal(), (-1, 1))	
			x_t[['x_prior']]=kf.x_prior
			x_t[['x_post ']]=kf.x_post	
			P_t[['P_t']]=np.reshape(kf.P.diagonal(), (-1, 1))	    
			P_t[['P_t_prior']]=np.reshape(kf.P_prior.diagonal(), (-1, 1))
			P_t[['P_t_post']]=np.reshape(kf.P_post.diagonal(), (-1, 1))
			Q_t[['Q']]=np.reshape(kf.Q.diagonal(), (-1,1))
			R_t[['R']]=np.reshape(kf.R.diagonal(), (-1,1))
			S_t[['S']]=np.reshape(kf.S.diagonal(), (-1,1))
			y_t[['y']]=kf.y
			input_output.append_state_history(dates[i-1].date(),x_t, m_t, K_t, P_t, Q_t, R_t, S_t, y_t, path)
			

	return kf, x_t


def predict_and_save(kf, x_t, path):
	kf.predict()
	x_t[['CR']]=kf.x
	input_output.write_kalman_state(kf.R, kf.Q, kf.P, kf.H, kf.F, x_t, path)
	return kf, x_t


def update_and_save(kf, m_t, x_t):
	#t+=1
	mean=m_t['Conversions'].sum()/(m_t['Clicks']+0.001).sum()
	m_t['Clicks']=m_t['Clicks'].apply(lambda x: math.nan if x==0 else x)
	cl=m_t['Clicks']+1
	co=m_t['Conversions']+mean
	r=np.abs(1/(cl)*(co/(cl*(cl-co)/(cl))))#**2 #=σ^2

	#if we get no click, R needs to be high
	r=r.fillna(1000)
	R_t=pd.DataFrame(data=np.diag(r),index=x_t.index, columns=x_t.index)
	kf.R=R_t.to_numpy()
	kf.update(z=m_t['CR'].to_numpy())
	return kf



def initiate_kalman_state(df_clustered):	
	############################################ Kalman Filter & Prediction ##########################################
	# We ássume the distribution is Gaussian and the problem is a binomial problem: either a click convert or not. 
	# We want to predict the CR for each keyword/biddable object.
	# We clustered first keywords in node and assume every keyword in a node have the same conversion rate (but not necessary the same fitting curve)

	# variance and sample:
	# https://keithmcnulty.github.io/bayes_analysis/global_water.html
	# https://courses.lumenlearning.com/introstats1/chapter/mean-or-expected-value-and-standard-deviation/#:~:text=To%20calculate%20the%20standard%20deviation%20(%CF%83)%20of%20a%20probability%20distribution,and%20take%20the%20square%20root.
	# the probability density function is given by:
	# P(C,N|p)=(C+N)!/(C!N!)*p^C*(1-p)^N where p is the probability
	# C the number of click which converted, N the number of clicks which didn't converted
	# https://docs.google.com/spreadsheets/d/1uFsp0xXAw9jiyzEgQhlBqGHef7RzQbnnBJgwmHDktzQ/edit#gid=1270622899
	# the mean and standard deviation can be calculated either by the sum product of the probability distribution or as follow:
	# μ=conversion/# of clicks
	# σ=(1/#clicks)*(#conversion/#clicks)*(#not conversion/#clicks)

	# Some info about Kalman Filter
	# https://www.youtube.com/watch?v=fpRb1sjFz_M
	# https://medium.com/@jaems33/understanding-kalman-filters-with-python-2310e87b8f48
	# https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
	# https://en.wikipedia.org/wiki/Recursive_Bayesian_estimation
	# https://stats.stackexchange.com/questions/183118/difference-between-hidden-markov-models-and-particle-filter-and-kalman-filter
	# http://ros-developer.com/2019/04/10/kalman-filter-explained-with-python-code-from-scratch/
	# https://stackoverflow.com/questions/43377626/how-to-use-kalman-filter-in-python-for-location-data

	# Z is our observation. We want Z to be a vector of the conversion rate for each node and for each keyword(leaf). 

	# Kalman variables:
	# X [1,n] is the state of the process (the conversion rate for each node)
	# M [1,n] is the measurment matrix
	# P [n,n] is the process covariance matrix, will be update by the Kalman filter. That the variance of the system.
	# Q [n,n] is the noise in the process.(covariance matrix associated with the noise in states)
	# R [n,n] is the measurement variance.
	# H [n,n] is the measurement function. Converting the measurement to the state so right now 1 because we have 1 state, 1 measure.	
	# F [n,n](also called A in the litterature) is the transition matrix to predict the next step based on the current state. 
	#         It can be either an identity matrix or predict from the parent-children relationship
	# B [1,n] is control transition matrix. The matrix models weekly seasonality
	# K [n,n] between 0 and 1: 1 we trust only measure, 0 we trust only the prediction
	# S [n,n] System uncertainty projected into measurement space
	####################################################################################################################################

	# get the previous CR estimate or take average to initiate.
	x_t=pd.DataFrame(index=df_clustered['Leave'].unique(), columns=['CR'])
	avgCR=df_clustered['Conversions'].sum() / df_clustered['Clicks'].sum()
	#x_t['CR'] = (df_clustered.reset_index().groupby('Leave')['Conversions'].sum()+avgCR) / (
	#df_clustered .reset_index().groupby('Leave')['Clicks'].sum()+1) 
	x_t['CR'] = avgCR
	return x_t

def set_index_date(df_clustered):
	# Initial State Matrix: we take the conversion rate of the 3 last months
	df_clustered['Date']=df_clustered['Date'].dt.date
	df_clustered=df_clustered.set_index('Date')
	df_clustered=df_clustered.sort_index()
	return df_clustered 

def get_kalman_measurment(x_t, df_clustered ):
	# M is the measurment or conversion rate for each node 
	# We could use unbiased conversion rate: 
	#conversion obeserved + 1 click * avg conversion rate / click observed + 1 click
	# Handling of missing data:
	# https://math.stackexchange.com/questions/982982/kalman-filter-with-missing-measurement-inputs
	# For a missing measurement, just use the last state estimate as a measurement but set 
	# the covariance matrix of the measurement to essentially infinity. 
	# This would cause a Kalman filter to essentially ignore the new measurement since 
	# the ratio of the variance of the prediction to the measurement is zero. 
	# The result will be a new prediction that maintains same state but 
	# whose variance will grow according to the process noise.
	print(df_clustered)
	try:
		mean= df_clustered['Conversions'].sum()/df_clustered['Clicks'].sum()
	except ZeroDivisionError:
		mean= (df_clustered['Conversions'].sum()+1)/(df_clustered['Clicks'].sum()+1)
	m_t=pd.DataFrame(index=x_t.index, columns=['Clicks', 'CR'])
	
	m_t['Clicks']=df_clustered.reset_index().groupby('Leave')['Clicks'].sum()
	m_t['Clicks']=m_t['Clicks'].apply(lambda x: math.nan if x==0 else x+4)
	m_t['Conversions']=(
		df_clustered.reset_index().groupby('Leave')['Conversions'].sum()+4*mean  
		)
	m_t['CR']=(m_t['Conversions']) / (m_t['Clicks'])
	m_t['CR']=m_t['CR'].fillna(x_t['CR'])
	return m_t

def create_kalman_filter(x_t):
	n=len(x_t)#.dropna())
	kf = KalmanFilter(dim_x=n, dim_z=n)
	return kf, n

def initiate_process_matrix(x_t, kf, n):	# Initial Estimation Covariance Matrix/Process Matrix. 
	P_t=pd.DataFrame(data=np.identity(n)*0.5, index=x_t.index, columns=x_t.index)
	kf.P *= 0.5 #2% variance in conversion rate
	return kf

def initiate_measurment_covariance(kf, m_t, x_t):	# Measurment covariance matrix
	#σ=(1/#clicks)*(#conversion/#clicks)*(#not conversion/#clicks)
	# If #conversion=0 σ=0 so we add average # of conversion
	mean=m_t['Conversions'].sum()/m_t['Clicks'].sum()
	m_t['Clicks']=m_t['Clicks'].apply(lambda x: math.nan if x==0 else x)
	r=abs(1/(m_t['Clicks']+1)*((m_t['Conversions']+mean)/(m_t['Clicks'])+1)*(1+m_t['Clicks']-m_t['Conversions']-mean)/(m_t['Clicks'])+1)#**2 #=σ^2
	#if we get no click, R needs to be high
	r=r.fillna(1000)
	R_t=pd.DataFrame(data=np.diag(r),index=x_t.index, columns=x_t.index)
	kf.R=R_t.to_numpy()
	return kf

def measurement_function(kf, n):
	kf.H=np.diag(np.ones(n))
	return kf

def initiate_process_covariance(kf, df_clustered):
	# with Q we can control how fast we allow the estimate to change. Higher Q means we trust less the process
	# cl=df_clustered.reset_index().groupby('Leave')['Clicks'].sum().to_numpy()
	# co=df_clustered.reset_index().groupby('Leave')['Conversions'].sum().to_numpy()
	# q=abs(1/(df_clustered['Clicks'])*(df_clustered['Conversions']/(df_clustered['Clicks']))*(df_clustered['Clicks']-df_clustered['Conversions'])/(df_clustered['Clicks']))
	cl=40
	co=40*df_clustered['Conversions'].sum()/df_clustered['Clicks'].sum()
	q=np.abs(1/(cl)*(co/(cl*(cl-co)/(cl))))
	kf.Q*=q
	#kf.Q*=0.2#np.diag(q)#0.01#np.diag(q) 
	return kf
	
def initiate_reg_to_mean_model(kf,n):
	A = np.diag(np.ones(n)) - np.diag(np.ones(n))*n/100 +np.ones((n,n))/100
	kf.F=A#np.diag(np.ones(n))
	return kf

def initiate_naive_model(kf,n):
	A = np.diag(np.ones(n))
	kf.F=A#np.diag(np.ones(n))
	return kf

def initiate_prophet_model(kf, n, df):
	A = np.diag(np.ones(n))
	kf.F=A#np.diag(np.ones(n))
	return kf

def get_conversion_rate(df, fromdate, untildate):

	df['Conv_Rate']=df.loc[fromdate:untildate]['Conversions'].sum().divide(
		df.loc[fromdate:untildate]['Clicks'].sum())	

def constant_sampling(df_clustered, look_back_window, i, dates):
	day_data = df_clustered.loc[dates[i-look_back_window:i]][['Leave','Conversions', 'Clicks']].groupby('Leave').sum()
	day_data['Leave'] = day_data.index
	day_data['Date'] = dates[i]
	day_data.set_index('Date', inplace = True)
	dic[dates[i]] = day_data
	print(f'From {dates[i-look_back_window].date()} to {dates[i-1].date()} (both inclusive)')
	print(day_data.set_index('Leave'))
	print('~'*100)
	return day_data

def auto_sampling(df_clustered, threshold, i, end_date):
	df_auto_sampling = pd.DataFrame(columns = ['Leave', 'Start date', 'End date', 'Range', 'Clicks', 'Conversions'])

	for leave in df_clustered.loc[:end_date].Leave.unique():
		df_leave = df_clustered.loc[:end_date].query('Leave == @leave').copy()
		flag, result = smallest_date_range(leave, df_leave, threshold, end_date)
		if flag:
			df_auto_sampling = df_auto_sampling.append(result)
			#df_auto_sampling = pd.concat([df_auto_sampling, result])
	df_auto_sampling['Start date'] = df_auto_sampling['Start date'].apply(lambda x: x.date())
	df_auto_sampling['End date'] = df_auto_sampling['End date'].apply(lambda x: x.date())
	return df_auto_sampling


def smallest_date_range(leave, df, threshold, end_date):
	df.loc[:,'Clicks_1D'] = df.loc[:,'Clicks'].rolling('1d').sum()
	df.loc[:,'Clicks_7D'] = df.loc[:,'Clicks'].rolling('7d').sum()
	df.loc[:,'Clicks_14D'] = df.loc[:,'Clicks'].rolling('14d').sum()
	df.loc[:,'Clicks_28D'] = df.loc[:,'Clicks'].rolling('28d').sum()
	df.loc[:,'Clicks_48D'] = df.loc[:,'Clicks'].rolling('48d').sum()
	df.loc[:,'Clicks_96D'] = df.loc[:,'Clicks'].rolling('96d').sum()
	df.loc[:,'Order_1D'] = df.loc[:,'Conversions'].rolling('1d').sum()
	df.loc[:,'Order_7D'] = df.loc[:,'Conversions'].rolling('7d').sum()
	df.loc[:,'Order_14D'] = df.loc[:,'Conversions'].rolling('14d').sum()
	df.loc[:,'Order_28D'] = df.loc[:,'Conversions'].rolling('28d').sum()
	df.loc[:,'Order_48D'] = df.loc[:,'Conversions'].rolling('48d').sum()
	df.loc[:,'Order_96D'] = df.loc[:,'Conversions'].rolling('96d').sum()
	result = pd.DataFrame(columns = ['Leave', 'Start date', 'End date', 'Range', 'Clicks'])
	#end_date = df.index[-1]
	dayrange = 0

	if df['Clicks_1D'][-1] >= threshold:
		dayrange = 1
		clickvalue = df['Clicks_1D'][-1]
		convvalue =   df['Order_1D'][-1]  
	elif df['Clicks_7D'][-1] >= threshold:
		dayrange = 7
		clickvalue = df['Clicks_7D'][-1]
		convvalue =   df['Order_7D'][-1]  
	elif df['Clicks_14D'][-1] >= threshold:
		dayrange = 14
		clickvalue = df['Clicks_14D'][-1]
		convvalue =   df['Order_14D'][-1]  
	elif df['Clicks_28D'][-1] >= threshold:
		dayrange = 28
		clickvalue = df['Clicks_28D'][-1]
		convvalue =   df['Order_28D'][-1]  
	elif df['Clicks_48D'][-1] >= threshold:
		dayrange = 48
		clickvalue = df['Clicks_48D'][-1]
		convvalue =   df['Order_48D'][-1]  
	else:
		dayrange = 96
		clickvalue = df['Clicks_96D'][-1]
		convvalue =   df['Order_96D'][-1]  

	start_date = end_date - datetime.timedelta(days=dayrange)
	#start_date = min([x for x in df.index if x > start_date and x <= end_date], default="EMPTY")

	s = pd.Series({'Leave' : leave, 'Start date' : start_date, 'End date' : end_date, 
						'Range' : dayrange, 'Clicks' : clickvalue, 'Conversions' : convvalue})

	result = result.append(s, ignore_index =True)
	#result = pd.concat([result, s] , ignore_index =True)
	return True, result


def get_last_measure(df_clustered,lastdayfrom,samplingdays):
	# Get the last measure given the # of day from selfsampling. 
	# This should return the data for each node with similar obsevation number (clicks). 
	lastdayfrom=date_helpers.date_ymd(lastdayfrom)
	df=df_clustered.loc[df_clustered.index<=lastdayfrom] #get the value until today
	df2=pd.DataFrame(columns=['Clicks','Conversions'])
	for leavedata in df.groupby('Leave'):
		leave=leavedata[0]
		date_begin=lastdayfrom-datetime.timedelta(samplingdays[[leave]].values[0])
		df2=df2.append(leavedata[1].loc[date_begin:].groupby('Leave').agg('sum'))
	df2['CR']=df2['Conversions']/df2['Clicks']
	return df2

if __name__ == "__main__":
	main()
