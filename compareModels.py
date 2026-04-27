import pickle

# after training each model:

pickle.dump(lr_model, open("saved_models/lr.pkl", "wb"))
pickle.dump(nb_model, open("saved_models/nb.pkl", "wb"))
pickle.dump(svm_model, open("saved_models/svm.pkl", "wb"))
pickle.dump(vectosrizer, open("saved_models/vectorizer.pkl", "wb"))