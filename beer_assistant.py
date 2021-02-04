#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: regan
"""

import csv
from scipy import optimize as opt
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import re

def get_beer_review_data(filename):
    file = open(filename);
    full_read = csv.reader(file);
    
    ans     = dict();
    headers = next(full_read);
    for category in headers:
        ans[category] = list();
    
    ncat = len(headers);
    for row in full_read:
        for j in range(0, ncat):
            ans[headers[j]].append(row[j]);

    file.close();
    return ans;

def get_beer_indices_from_most_to_least_reviewed():
    
    filename = "beer_index_from_most_to_least_reviewed.pkl";
    if os.path.isfile(filename):
        infile = open(filename, 'rb');
        ans = pickle.load(infile);
        infile.close();
        return ans;
    
    beer_reviews = get_beer_review_data('beer_reviews.csv');
    
    beer_index = dict();
    # first count numbers of reviews for each item:
    for beer in beer_reviews['beer_name']:
        beer_index[beer] = beer_index.get(beer, 0) + 1;
    
    # sort by decreasing number of reviews
    beer_index = dict(sorted(beer_index.items(), key=lambda item: item[1], reverse=True));
    
    # tag items:
    idx = 0;
    for beer in beer_index:
        beer_index[beer] = idx;
        idx += 1;
        
    outfile = open(filename, 'wb');
    pickle.dump(beer_index, outfile, pickle.HIGHEST_PROTOCOL);
    outfile.close()
    
    return beer_index;

def get_reviewer_indices_from_most_to_least_active(nmost_popular_beers):
    
    filename = "reviewer_index_from_most_to_least_active_on_" + str(nmost_popular_beers) + "_most_reviewed_beers.pkl";
    if os.path.isfile(filename):
        infile = open(filename, 'rb');
        ans = pickle.load(infile);
        infile.close();
        return ans;
    
    beer_reviews    = get_beer_review_data('beer_reviews.csv');
    beer_index      = get_beer_indices_from_most_to_least_reviewed();
    reviewer_index  = dict();
    # first count numbers of reviews for each qualifying item:
    for i in range(0, len(beer_reviews['beer_name'])):
        beer_id     = beer_index[beer_reviews['beer_name'][i]];
        if beer_id >= nmost_popular_beers:
            continue;
            
        reviewer_name = beer_reviews['review_profilename'][i];
        reviewer_index[reviewer_name] = reviewer_index.get(reviewer_name, 0) + 1;
    
    # sort by decreasing number of reviews
    reviewer_index = dict(sorted(reviewer_index.items(), key=lambda item: item[1], reverse=True));
    
    # tag items:
    idx = 0;
    for reviewer in reviewer_index:
        reviewer_index[reviewer] = idx;
        idx += 1;
        
    outfile = open(filename, 'wb');
    pickle.dump(reviewer_index, outfile, pickle.HIGHEST_PROTOCOL);
    outfile.close()
    
    return reviewer_index;


def get_ratings_matrices(beer_reviews, nbeers, training_size, cv_size):
    
    beer_id_for     = get_beer_indices_from_most_to_least_reviewed();
    reviewer_id_for = get_reviewer_indices_from_most_to_least_active(nbeers);
    
    if training_size + cv_size > len(reviewer_id_for):
        raise Exception("Your desired training size", training_size , " and cross validation size", cv_size, "is larger than the total datasets of", len(reviewer_id_for), "reviewers for the", nbeers, "most reviewed beers");

    training_ratings        = np.zeros((nbeers, training_size));
    training_is_rated       = np.zeros((nbeers, training_size));
    validation_ratings      = np.zeros((nbeers, cv_size));
    validation_is_rated     = np.zeros((nbeers, cv_size));
    
    rating_category = 'review_overall';
    for i in range(0, len(beer_reviews[rating_category])):
        beer_id     = beer_id_for[beer_reviews['beer_name'][i]];
        if beer_id >= nbeers:
            continue;
        reviewer_id = reviewer_id_for[beer_reviews['review_profilename'][i]];
        if reviewer_id > training_size + cv_size:
            continue;
        if reviewer_id < training_size:
            training_ratings[beer_id][reviewer_id]  = beer_reviews[rating_category][i];
            training_is_rated[beer_id][reviewer_id] = 1.0;
        if training_size <= reviewer_id and reviewer_id < training_size + cv_size:
            validation_ratings[beer_id][reviewer_id - training_size]  = beer_reviews[rating_category][i];
            validation_is_rated[beer_id][reviewer_id - training_size] = 1.0;

    return training_ratings, training_is_rated, validation_ratings, validation_is_rated;

def get_rating_error(item_features, user_offsets, user_tastes, ratings, is_rated):
    return  np.multiply(np.tensordot(item_features, user_tastes, axes=([1],[1])) + user_offsets - ratings, is_rated);

def cost_function(parameters, ratings, is_rated, n_features, regularizer):
    
    n_items         = ratings.shape[0];
    n_users         = ratings.shape[1];
    
    item_features   = parameters[0:n_items*n_features].reshape((n_items, n_features));
    user_offsets    = parameters[n_items*n_features:n_items*n_features + n_users]; # user offsets
    user_tastes     = parameters[n_items*n_features + n_users:].reshape((n_users, n_features));

    rating_error    = get_rating_error(item_features, user_offsets, user_tastes, ratings, is_rated);
    
    cost = 0.5*(np.sum(rating_error**2) + regularizer*(np.sum(item_features**2) + np.sum(user_tastes**2)));
    
    return cost;


def gradient(parameters, ratings, is_rated, n_features, regularizer):
    
    n_items         = ratings.shape[0];
    n_users         = ratings.shape[1];
    
    item_features   = parameters[0:n_items*n_features].reshape((n_items, n_features));
    user_offsets    = parameters[n_items*n_features:n_items*n_features + n_users]; # user offsets
    user_tastes     = parameters[n_items*n_features + n_users:].reshape((n_users, n_features));
    
    rating_error    = get_rating_error(item_features, user_offsets, user_tastes, ratings, is_rated);
    
    grad_item_features  = np.matmul(rating_error, user_tastes) + regularizer*item_features;
    grad_user_offsets   = np.sum(rating_error,axis=0);
    grad_user_tastes    = np.tensordot(rating_error, item_features, axes=([0],[0])) + regularizer*user_tastes;
    
    grad = np.concatenate((grad_item_features.reshape(n_items*n_features), grad_user_offsets, grad_user_tastes.reshape(n_users*n_features)));
    
    return grad;

def get_optimal_parameters(ratings, is_rated, n_features, regularizer):
    parameters = np.random.rand(ratings.shape[0]*n_features + ratings.shape[1]*(1 + n_features));
    return opt.fmin_cg(cost_function, parameters, fprime=gradient, args=(ratings, is_rated, n_features, regularizer));

def get_training_parameters_for(n_first_beers, training_size, nfeatures, regularizer):
    
    directory = str(nfeatures) + "_features_with_lambda_" + str(round(regularizer, 3));
    filename = directory + "/training_parameters_for_" + str(n_first_beers) + "_first_beers_trained_with_" + str(training_size) + "_first_users.npy";
    if os.path.isfile(filename):
        infile = open(filename, 'rb');
        ans = np.load(infile);
        infile.close();
        return ans;
    
    beer_reviews    = get_beer_review_data('beer_reviews.csv');
    training_ratings, training_is_rated,_,_ = get_ratings_matrices(beer_reviews, n_first_beers, training_size, 0);
    optimal_parameters = get_optimal_parameters(training_ratings, training_is_rated, nfeatures, regularizer);
    
    if not os.path.isdir(directory):
        os.mkdir(directory);
    outfile = open(filename, 'wb');
    np.save(outfile, optimal_parameters);
    outfile.close()
    
    return optimal_parameters;

def cost_function_new_users(user_parameters, ratings, is_rated, item_features, regularizer):
    
    n_users         = ratings.shape[1];
    n_features      = item_features.shape[1];
    
    user_offsets    = user_parameters[0:n_users]; # user offsets
    user_tastes     = user_parameters[n_users:].reshape((n_users, n_features));
    
    rating_error    = get_rating_error(item_features, user_offsets, user_tastes, ratings, is_rated);
    
    cost = 0.5*(np.sum(rating_error**2) + regularizer*np.sum(user_tastes**2)); # dropped regularizer*np.sum(item_features**2) because it's "constant" in this case
    
    return cost;

def gradient_new_users(user_parameters, ratings, is_rated, item_features, regularizer):
    
    n_users         = ratings.shape[1];
    n_features      = item_features.shape[1];
    
    user_offsets    = user_parameters[0:n_users]; # user offsets
    user_tastes     = user_parameters[n_users:].reshape((n_users, n_features));
    
    rating_error    = get_rating_error(item_features, user_offsets, user_tastes, ratings, is_rated);
    
    grad_user_offsets   = np.sum(rating_error,axis=0);
    grad_user_tastes    = np.tensordot(rating_error, item_features, axes=([0],[0])) + regularizer*user_tastes;
    
    grad = np.concatenate((grad_user_offsets, grad_user_tastes.reshape(n_users*n_features)));
    
    return grad;

def get_user_parameters(ratings, is_rated, item_features, regularizer):
    parameters = np.random.rand(ratings.shape[1]*(1 + item_features.shape[1]));
    return opt.fmin_cg(cost_function_new_users, parameters, fprime=gradient_new_users, args=(ratings, is_rated, item_features, regularizer));

def get_cross_validation_user_parameters_for(n_first_beers, training_size, cv_size, nfeatures, regularizer):
    
    directory = str(nfeatures) + "_features_with_lambda_" + str(round(regularizer, 3));
    filename = directory + "/cv_user_parameters_for_cv_size_of_" + str(cv_size) +"_users_for_features_learned_on_" + str(n_first_beers) + "_first_beers_trained_with_" + str(training_size) + "_first_users.npy";
    if os.path.isfile(filename):
        infile = open(filename, 'rb');
        ans = np.load(infile);
        infile.close();
        return ans;
    
    beer_reviews    = get_beer_review_data('beer_reviews.csv');
    
    training_parameters = get_training_parameters_for(n_first_beers, training_size, nfeatures, regularizer);
    beer_features       = training_parameters[0:n_first_beers*nfeatures].reshape((n_first_beers, nfeatures));
    _,_, cv_ratings, cv_is_rated = get_ratings_matrices(beer_reviews, n_first_beers, training_size, cv_size);
    
    optimal_parameters  = get_user_parameters(cv_ratings, cv_is_rated, beer_features, regularizer);
    
    if not os.path.isdir(directory):
        os.mkdir(directory);
    outfile = open(filename, 'wb');
    np.save(outfile, optimal_parameters);
    outfile.close()
    
    return optimal_parameters;

def get_training_and_cv_cost(n_first_beers, training_size, cv_size, nfeatures, regularizer):
    beer_reviews        = get_beer_review_data('beer_reviews.csv');
    
    training_ratings, training_is_rated, cv_ratings, cv_is_rated = get_ratings_matrices(beer_reviews, n_first_beers, training_size, cv_size);
    training_parameters = get_training_parameters_for(n_first_beers, training_size, nfeatures, regularizer);
    cv_user_parameters  = get_cross_validation_user_parameters_for(n_first_beers, training_size, cv_size, nfeatures, regularizer);
    beer_features       = training_parameters[0:n_first_beers*nfeatures].reshape((n_first_beers, nfeatures));
    training_u_offsets  = training_parameters[n_first_beers*nfeatures:n_first_beers*nfeatures + training_size];
    training_u_tastes   = training_parameters[n_first_beers*nfeatures + training_size:].reshape((training_size, nfeatures));
    
    training_rating_error = get_rating_error(beer_features, training_u_offsets, training_u_tastes, training_ratings, training_is_rated);
    n_rated_training    = np.sum(training_is_rated);
    training_rmse       = np.sqrt(np.sum(training_rating_error**2)/n_rated_training);
    
    cv_rating_error     = get_rating_error(beer_features, cv_user_parameters[0:cv_size], cv_user_parameters[cv_size:].reshape((cv_size, nfeatures)), cv_ratings, cv_is_rated);
    n_rated_cv          = np.sum(cv_is_rated);
    cv_rmse             = np.sqrt(np.sum(cv_rating_error**2)/n_rated_cv);
    
    training_cost       = cost_function(training_parameters, training_ratings, training_is_rated, nfeatures, regularizer);
    cv_cost             = cost_function_new_users(cv_user_parameters, cv_ratings, cv_is_rated, beer_features, regularizer);
    return training_cost, cv_cost, training_rmse, cv_rmse, n_rated_training, n_rated_cv;


def plot_training_and_cv_rmse_and_mean_cost_function_for(nbeers, nfeatures, training_size, cv_size, regularizer_min, regularizer_max, regularizer_increment):
    regularizer         = regularizer_min if regularizer_increment > 0.0 else regularizer_max;
    if regularizer <= 0.0:
        raise Exception("plot_training_and_cv_rmse_and_mean_cost_function_for: The regularizer hyper-parameter must be strictly positive");
        
    regularizer_values  = np.array([]);
    cv_mean_cost        = np.array([]);
    training_mean_cost  = np.array([]);
    cv_rmse             = np.array([]);
    training_rmse       = np.array([]);
    
    while ((regularizer >= regularizer_min) if regularizer_increment < 0.0 else ((regularizer <= regularizer_max))):
        regularizer = max(regularizer, 0.01);
        regularizer = round(regularizer, 3)
        training_cost, cv_cost, training_rmse_, cv_rmse_, n_rated_training, n_rated_cv = get_training_and_cv_cost(nbeers, nusers_training, nusers_cv, nfeatures, regularizer)
        regularizer_values  = np.append(regularizer_values, regularizer);
        training_mean_cost  = np.append(training_mean_cost, training_cost/n_rated_training);
        cv_mean_cost        = np.append(cv_mean_cost, cv_cost/n_rated_cv);
        training_rmse       = np.append(training_rmse, training_rmse_);
        cv_rmse             = np.append(cv_rmse, cv_rmse_);
        regularizer += regularizer_increment;
        
    plt.plot(regularizer_values, cv_rmse, 'r-o', label="RMSE on cv set of users")
    plt.xlabel('Regularizer')
    plt.plot(regularizer_values, training_rmse, 'b-o', label="RMSE on training set of users")
    plt.legend(loc='best')
    plt.savefig('RMSE_cv_and_training.png')
    plt.show()
    
    plt.plot(regularizer_values, cv_mean_cost, 'r-o', label="Optimal (mean) cost function on cv set of users")
    plt.xlabel('Regularizer')
    plt.plot(regularizer_values, training_mean_cost, 'b-o', label="Optimal (mean) cost function on training set of users")
    plt.legend(loc='best')
    plt.savefig('optimal_mean_cost_function_cv_and_training.png')
    plt.show()
    
    return;
    
def get_beer_idx(beer_name, idx_max):
    beer_index = get_beer_indices_from_most_to_least_reviewed();
    
    candidates = [];
    for beer_candidate in beer_index:
        if re.search(beer_name, beer_candidate, re.IGNORECASE):
            candidates.append(beer_candidate);
    
    if not candidates:
        print("Unfortunately, there is no match for that beer name in my data base...")
        return -1;
    
    if len(candidates) == 1:
        print("I found this possible match in my data_base:", candidates[0]);
        answer = input("Is it what you had in mind?[yes/y/no/n] ");
        if answer.lower() == "yes" or answer.lower() == "y":
            return beer_index[candidates[0]];
        else:
            print("I am sorry I am afraid I am not aware of that beer...")
            return -1;
    
    list_idx = 0;
    print("I found these possible matches in my data_base, which one did you have in mind?")
    for element in candidates:
        print("Enter", list_idx, "for", element);
        list_idx += 1;
    print("[if none of the above matches your thought, enter any other number]")
    list_idx_input = int(input("Which one did you mean? "));
    
    if list_idx_input < 0 or list_idx_input >= len(candidates):
        print("[sorry to disappoint :-(]")
        return -1;
    
    return beer_index[candidates[list_idx_input]];
    
def get_to_know_your_user(nbeers, nfeatures):
    
    
    user_ratings    = np.zeros((nbeers, 1));
    user_has_rated  = np.zeros((nbeers, 1));
    
    print("I would need to know you and your tastes better before I can make")
    print("any recommendation: please take some time to think about beers")
    print("you already know and let me know how much you like/dislike them,")
    print("on a scale from 1 to 5. In order to give sensible recommendations,")
    print("I need you to share your feelings about", nfeatures, "beers, at least")
    print("[Since I am currenlty in training, I may not know some beer you")
    print("mention and I may ask you to find other ones if possible. ")
    print("Thank you for your understanding ;-)]")
    
    nbeers_shared = 0;
    quit_instruction = "QQQ"
    beer_name_instruction = "Tell me the name of a beer you know (or type " + quit_instruction + " if you are done): ";
    rating_instruction = "How much did you like it (on a scale from 1 to 5): ";
    beer_name = input(beer_name_instruction);
    while beer_name.lower() != quit_instruction.lower():
        beer_id = get_beer_idx(beer_name, nbeers);
        if beer_id >= nbeers:
            print("Actually, I did not include that beer in my training yet, unfortunately: thank you for your patience...")
        if beer_id >= 0 and beer_id < nbeers:
            
            if user_has_rated[beer_id] < 0.5:
                rating = float(input(rating_instruction));
                while rating < 1.0 or rating > 5.0:
                    print("I need this rating to be between 1 and 5, please do it again.")
                    rating = float(input(rating_instruction));
                user_ratings[beer_id] = rating;
                user_has_rated[beer_id] = 1;
                nbeers_shared += 1
            else:
                print("Well, you already told me about that one, I expect another one...")
                
        beer_name = input(beer_name_instruction);
    
    if nbeers_shared < nfeatures:
        raise Exception("This exception was thrown by your beer assistant and recommender system because you have not shared enough information for it to do a good job");

    return user_ratings, user_has_rated;

def make_prediction(user_parameters, beer_features):
    nfeatures = beer_features.shape[1];
    if len(user_parameters)%(1 + nfeatures) != 0:
        raise Exception("Can't make a prediction, user parameters do not math beer features...");
    nusers = len(user_parameters)//(1 + nfeatures);
    user_offset = user_parameters[0:nusers];
    user_taste  = user_parameters[nusers:].reshape(nusers, nfeatures);
    return np.tensordot(beer_features, user_taste, axes=([1],[1])) + user_offset;

def make_suggestions(user_ratings, user_has_rated, beer_features, regularizer):
    print("-------------------------------------------------------------------------------------------")
    print("|                                                                                         |")
    print("Thank you for your input: I will try to figure out your preferences (give me a bit of time)")
    print("|                                                                                         |")
    print("-------------------------------------------------------------------------------------------")
    user_parameters = get_user_parameters(user_ratings, user_has_rated, beer_features, regularizer);
    predicted_ratings = make_prediction(user_parameters, beer_features);
    ranked_suggestions = sorted(range(len(predicted_ratings)), key=lambda k: predicted_ratings[k], reverse=True);
    beer_index = get_beer_indices_from_most_to_least_reviewed();
    suggestion_idx = 0;
    running_idx = 0;
    print("I recommend you to check out the following ones")
    while suggestion_idx < nsuggestions :
        if user_has_rated[ranked_suggestions[running_idx]] < 0.5: # i.e., it wasn't given by the user in the first place
            suggestion_idx += 1;
            prediction = predicted_ratings[ranked_suggestions[running_idx]];
            if prediction <= 5.0 and prediction >= 1.0:
                print("Suggestion #", suggestion_idx, ":", list(beer_index)[ranked_suggestions[running_idx]], " (my predicted rating is", prediction, ")");
            elif prediction > 5.0:
                print("Suggestion #", suggestion_idx, ":", list(beer_index)[ranked_suggestions[running_idx]], " (my predicted rating is > 5)");
            else:
                print("Suggestion #", suggestion_idx, ":", list(beer_index)[ranked_suggestions[running_idx]], " (my predicted rating is < 1)");
        running_idx += 1;
    return;

nbeers              = 2000;
nusers_training     = 1000; # number of users considered for training (collaborative filtering)
nusers_cv           =  200; # number of users considered for cross-validation (hypertuning of regularization factor)
nfeatures           =    5; # though this should be considered a hyper-parameter, we will not touch that one, for now
nsuggestions        =   10;

#plot_training_and_cv_rmse_and_mean_cost_function_for(nbeers, nfeatures, nusers_training, nusers_cv, 0., 0.6, -0.05);

regularizer = 0.15;
print("I am considering a collaborative filtering strategy, trained on the")
print(nbeers, "most reviewed beers in the data base and the", nusers_training, "most active")
print("reviewer profiles considering these beers. I give", nfeatures, "features per")
print("beer and I use a regularization parameter of", regularizer)

print("-------------------------------------------------------------------------------------------")
print("|                                                                                         |")
print("|                Hello, I am your beer assistant and recommender system                   |")
print("|                     Let's find new beers for you to check out!                          |")
print("|                                                                                         |")
print("-------------------------------------------------------------------------------------------")

training_parameters = get_training_parameters_for(nbeers, nusers_training, nfeatures, regularizer);
beer_features = training_parameters[0:nbeers*nfeatures].reshape((nbeers, nfeatures));
user_ratings, user_has_rated = get_to_know_your_user(nbeers, nfeatures);
make_suggestions(user_ratings, user_has_rated, beer_features, regularizer)
print("-------------------------------------------------------------------------------------------")
print("|                                                                                         |")
print("|            Thank you for using me, I hope to see you again soon!                        |")
print("|                                                                                         |")
print("-------------------------------------------------------------------------------------------")


