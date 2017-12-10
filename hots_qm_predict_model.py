import keras
from keras import backend as K
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from keras.regularizers import l2
import pandas as pd
from keras.utils.data_utils import get_file
import numpy as np

path = 'https://github.com/CommonGibbon/HOTS_QM_Predict/releases/download/v1.0_12-4-17/'
wname = 'qm_weights.h5'
id_path = path+'HeroIDAndMapID.csv'
norm_path = path + 'norm_params.csv'

def process_input(hero_names, player_hero_levels, player_mmrs,map_name):
    # This function is necessary since the hero names must be mapped to their IDs,
    # the level and mmr data must be normalized, and the map data must be one-hot encoded
    
    # create an dictionary which maps hero and map names to their respective IDs
    id_map = {}
    id_df = pd.read_csv(id_path)
    num_maps = id_df.loc[len(id_df)-1,"ID"] - 1000
    for _,row in id_df.iterrows():
        if row.ID >1000: # if its a map name
            id_map[row.Name] = row.ID-1001 # match map name to the augmented index
        else: # if its a hero name
            id_map[row.Name] = row.ID

    # pull the mean and standard deviation from github.
    norm_params_df = pd.read_csv(norm_path,names = ["lvl mean","lvl std","mmr mean","mmr std"])
    lvl_mean = norm_params_df["lvl mean"]
    lvl_std = norm_params_df["lvl std"]
    mmr_mean = norm_params_df["mmr mean"]
    mmr_std = norm_params_df["mmr std"]
    
    # Get the IDs
    hero_ids = [id_map[hero_names[i]] for i in range(5)]
    player_hero_levels_normed = [(player_hero_levels[i] - lvl_mean)/lvl_std for i in range(5)]
    player_mmrs_normed = [(player_mmrs[i] - mmr_mean)/mmr_std for i in range(5)]
    map_id = id_map[map_name]
    
    # one-hot the map ID
    enc = OneHotEncoder(n_values = num_maps)
    map_enc = enc.fit_transform(map_id).todense()
    
    # thess arrays can be passed as input to the model itself
    return np.array(hero_ids).reshape(1,5), np.array(player_hero_levels_normed).reshape(1,5), np.array(player_mmrs_normed).reshape(1,5), np.array(map_enc)
    

def hots_qm():
    # read in the number of maps from the id file; needed for the map encoding
    id_df = pd.read_csv(id_path)
    num_maps = id_df.loc[len(id_df)-1,"ID"] - 1000
    num_heros = num_heros = max(id_df.ID[id_df.ID<1000])-1 # subtract 1 because hero IDs start at 1

    # define symbolic inputs
    mmr_in = Input(shape=(5,),name = 'mmr_in')
    hLevel_in = Input(shape=(5,),name = 'hLevel')
    map_in = Input(shape=(num_maps,),name = 'map_in')
    queue_in = Input(shape=(4,),name = 'queue_in')
    
    n_factors = 16
    hero_in = Input(shape=(5,),dtype='int64',name='hero_in')
    h =  Embedding(num_heros,n_factors,input_length=5,embeddings_regularizer=l2(1e-4))(hero_in)
    
    p = 0.4 # This is the level of dropout I found to be most effective
    
    # define layers
    x = h
    x = Flatten()(x) # flatten out the embeddings
    x = Concatenate()([x,hLevel_in,mmr_in])
    x = Dense(90, activation = 'relu')(x)
    x = Dropout(p/5)(x)
    x = Concatenate()([x,map_in])
    x = Dense(90, activation = 'relu')(x)
    x = Dropout(p)(x)
    x = Dense(1,activation = 'sigmoid')(x)
    nn = Model([hero_in,hLevel_in,mmr_in,map_in],x, name='hots_qm')
    nn.compile(Adam(0.001),loss='binary_crossentropy',metrics = ['accuracy'])

    # load weights
    wp =get_file(wname, path+wname, cache_subdir='models')
    nn.load_weights(wp)
    
    return nn