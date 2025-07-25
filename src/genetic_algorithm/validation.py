'''
Logan Kelsch - 7/21/25
'''



def val_adversarial(
    X_train, X_test, std_out:bool=True
):
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    X_tv = np.vstack([X_train, X_test])
    y_tv = np.array([0]*len(X_train) + [1]*len(X_test))
    adv = LogisticRegression().fit(X_tv, y_tv)
    score = adv.score(X_tv, y_tv)
    if(score < 0.6):
        performance = 'GOOD'
    else:
        performance = 'FAIR'
    
    if(std_out):
        print("AUC distinguishing train vs test:", score, performance)

    return score
