# -*- coding: utf-8 -*-
from __future__ import division
import math
import operator as op
from functools import reduce
from .pycm_interpret import *


def BCD_calc(TOP, P, AM):
    '''
    This function calculate BCD (Bray–Curtis dissimilarity)
    :param TOP: test outcome positive
    :type TOP : dict
    :param P: condition positive
    :type P : dict
    :param AM: Automatic/Manual
    :type AM : int
    :return: BCD as float
    '''
    try:
        TOP_sum = sum(TOP.values())
        P_sum = sum(P.values())
        return abs(AM) / (P_sum + TOP_sum)
    except Exception:
        return "None"


def AM_calc(TOP, P):
    '''
    This function calculate AM (Automatic/Manual)
    :param TOP: test outcome positive
    :type TOP : int
    :param P: condition positive
    :type P : int
    :return: AM as int
    '''
    try:
        return TOP - P
    except Exception:
        return "None"


def lift_calc(PPV, PRE):
    '''
    This function calculate lift score
    :param PPV:  precision or positive predictive value
    :type PPV : float
    :param PRE: Prevalence
    :type PRE : float
    :return: lift score as float
    '''
    try:
        return PPV / PRE
    except Exception:
        return "None"


def GI_calc(AUC):
    '''
    This function calculate Gini index
    :param AUC: AUC (Area under the ROC curve)
    :type AUC: float
    :return: Gini index as float
    '''
    try:
        return 2 * AUC - 1
    except Exception:
        return "None"


def DP_calc(TPR, TNR):
    '''
    This function calculate DP (Discriminant power)
    :param TNR: specificity or true negative rate
    :type TNR : float
    :param TPR: sensitivity, recall, hit rate, or true positive rate
    :type TPR : float
    :return: DP as float
    '''
    try:
        X = TPR / (1 - TPR)
        Y = TNR / (1 - TNR)
        return (math.sqrt(3) / math.pi) * (math.log(X, 10) + math.log(Y, 10))
    except Exception:
        return "None"


def RCI_calc(mutual_information, reference_entropy):
    '''
    This function calculate RCI (Relative classifier information)
    :param mutual_information: mutual information
    :type mutual_information : float
    :param reference_entropy: reference entropy
    :type reference_entropy : float
    :return:  RCI as float
    '''
    try:
        return mutual_information / reference_entropy
    except Exception:
        return "None"


def dInd_calc(TNR, TPR):
    '''
    This function calculate dInd (Distance index)
    :param TNR: specificity or true negative rate
    :type TNR : float
    :param TPR: sensitivity, recall, hit rate, or true positive rate
    :type TPR : float
    :return: dInd as float
    '''
    try:
        result = math.sqrt(((1 - TNR)**2) + ((1 - TPR)**2))
        return result
    except Exception:
        return "None"


def sInd_calc(dInd):
    '''
    This function calculate sInd (Similarity index)
    :param dInd: dInd
    :type dInd : float
    :return: sInd as float
    '''
    try:
        return 1 - (dInd / (math.sqrt(2)))
    except Exception:
        return "None"


def AUNP_calc(classes, P, POP, AUC_dict):
    '''
    This function calculate AUNP
    :param classes: classes
    :type classes : list
    :param P: condition positive
    :type P : dict
    :param POP: population
    :type POP : dict
    :param AUC_dict: AUC (Area under the ROC curve) for each class
    :type AUC_dict : dict
    :return: AUNP as float
    '''
    try:
        result = 0
        for i in classes:
            result += (P[i] / POP[i]) * AUC_dict[i]
        return result
    except Exception:
        return "None"


def AUC_calc(TNR, TPR):
    '''
    This function calculate AUC (Area under the ROC curve for each class)
    :param TNR: specificity or true negative rate
    :type TNR : float
    :param TPR: sensitivity, recall, hit rate, or true positive rate
    :type TPR : float
    :return: AUC as float
    '''
    try:
        return (TNR + TPR) / 2
    except Exception:
        return "None"


def CBA_calc(classes, table, TOP, P):
    '''
    This function calculate CBA (Class balance accuracy)
    :param classes: classes
    :type classes : list
    :param table: input matrix
    :type table : dict
    :param TOP: test outcome positive
    :type TOP : dict
    :param P: condition positive
    :type P : dict
    :return: CBA as float
    '''
    try:
        result = 0
        class_number = len(classes)
        for i in classes:
            result += ((table[i][i]) / (max(TOP[i], P[i])))
        return result / class_number
    except Exception:
        return "None"


def RR_calc(classes, TOP):
    '''
    This function calculate RR (Global Performance Index)
    :param classes: classes
    :type classes : list
    :param TOP: test outcome positive
    :type TOP : dict
    :return: RR as float
    '''
    try:
        class_number = len(classes)
        result = sum(list(TOP.values()))
        return result / class_number
    except Exception:
        return "None"


def overall_MCC_calc(classes, table, TOP, P):
    '''
    This function calculate Overall_MCC
    :param classes: classes
    :type classes : list
    :param table: input matrix
    :type table : dict
    :param TOP: test outcome positive
    :type TOP : dict
    :param P: condition positive
    :type P : dict
    :return:  Overall_MCC as float
    '''
    try:
        cov_x_y = 0
        cov_x_x = 0
        cov_y_y = 0
        matrix_sum = sum(list(TOP.values()))
        for i in classes:
            cov_x_x += TOP[i] * (matrix_sum - TOP[i])
            cov_y_y += P[i] * (matrix_sum - P[i])
            cov_x_y += (table[i][i] * matrix_sum - P[i] * TOP[i])
        return cov_x_y / (math.sqrt(cov_y_y * cov_x_x))
    except Exception:
        return "None"


def CEN_misclassification_calc(
        table,
        TOP,
        P,
        i,
        j,
        subject_class,
        modified=False):
    '''
    This function calculate misclassification probability of classifying
    :param table: input matrix
    :type table : dict
    :param TOP: test outcome positive
    :type TOP : int
    :param P: condition positive
    :type P : int
    :param i: table row index (class name)
    :type i : any valid type
    :param j: table col index (class name)
    :type j : any valid type
    :param subject_class: subject to class (class name)
    :type subject_class: any valid type
    :param modified : modified mode flag
    :type modified : bool
    :return: misclassification probability of classifying as float
    '''
    try:
        result = TOP + P
        if modified:
            result -= table[subject_class][subject_class]
        result = table[i][j] / result
        return result
    except Exception:
        return "None"


def CEN_calc(classes, table, TOP, P, class_name, modified=False):
    '''
    This function calculate CEN (Confusion Entropy)/ MCEN(Modified Confusion Entropy)
    :param classes: classes
    :type classes : list
    :param table: input matrix
    :type table : dict
    :param TOP: test outcome positive
    :type TOP : int
    :param P: condition positive
    :type P : int
    :param class_name: reviewed class name
    :type class_name : any valid type
    :param modified : modified mode flag
    :type modified : bool
    :return: CEN(MCEN) as float
    '''
    try:
        result = 0
        class_number = len(classes)
        for k in classes:
            if k != class_name:
                P_j_k = CEN_misclassification_calc(
                    table, TOP, P, class_name, k, class_name, modified)
                P_k_j = CEN_misclassification_calc(
                    table, TOP, P, k, class_name, class_name, modified)
                if P_j_k != 0:
                    result += P_j_k * math.log(P_j_k, 2 * (class_number - 1))
                if P_k_j != 0:
                    result += P_k_j * math.log(P_k_j, 2 * (class_number - 1))
        if result != 0:
            result = result * (-1)
        return result
    except Exception:
        return "None"


def convex_combination(classes, TP, TOP, P, class_name, modified=False):
    '''
    This function calculate Overall_CEN coefficient
    :param classes: classes
    :type classes : list
    :param TP: true Positive Dict For All Classes
    :type TP : dict
    :param TOP: test outcome positive
    :type TOP : dict
    :param P: condition positive
    :type P : dict
    :param class_name: reviewed class name
    :type class_name : any valid type
    :param modified : modified mode flag
    :type modified : bool
    :return: coefficient as float
    '''
    try:
        class_number = len(classes)
        alpha = 1
        if class_number == 2:
            alpha = 0
        matrix_sum = sum(list(TOP.values()))
        TP_sum = sum(list(TP.values()))
        up = TOP[class_name] + P[class_name]
        down = 2 * matrix_sum
        if modified:
            down -= (alpha * TP_sum)
            up -= TP[class_name]
        return up / down
    except Exception:
        return "None"


def overall_CEN_calc(classes, TP, TOP, P, CEN_dict, modified=False):
    '''
    This function calculate Overall_CEN (Overall confusion entropy)
    :param classes: classes
    :type classes : list
    :param TP: true positive dict for all classes
    :type TP : dict
    :param TOP: test outcome positive
    :type TOP : dict
    :param P: condition positive
    :type P : dict
    :param CEN_dict: CEN dictionary for each class
    :type CEN_dict : dict
    :param modified : modified mode flag
    :type modified : bool
    :return: Overall_CEN(MCEN) as float
    '''
    try:
        result = 0
        for i in classes:
            result += (convex_combination(classes, TP, TOP, P, i, modified) *
                       CEN_dict[i])
        return result
    except Exception:
        return "None"


def IS_calc(TP, FP, FN, POP):
    '''
    This function calculate IS (Information score)
    :param TP: true positive
    :type TP : int
    :param FP: false positive
    :type FP : int
    :param FN: false negative
    :type FN : int
    :param POP: population
    :type POP : int
    :return: IS as float
    '''
    try:
        result = -math.log(((TP + FN) / POP), 2) + \
            math.log((TP / (TP + FP)), 2)
        return result
    except Exception:
        return "None"


def ncr(n, r):
    '''
    This function calculate n choose r
    :param n: n
    :type n : int
    :param r: r
    :type r :int
    :return: n choose r as int
    '''
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


def p_value_calc(TP, POP, NIR):
    '''
    This function calculate p_value
    :param TP: true positive
    :type TP : dict
    :param POP: population
    :type POP : int
    :param NIR: no information rate
    :type NIR : float
    :return: p_value as float
    '''
    try:
        n = POP
        x = sum(list(TP.values()))
        p = NIR
        result = 0
        for j in range(x):
            result += ncr(n, j) * (p ** j) * ((1 - p) ** (n - j))
        return 1 - result
    except Exception:
        return "None"


def NIR_calc(P, POP):
    '''
    This function calculate NIR (No information rate)
    :param P: condition positive
    :type P : dict
    :param POP: population
    :type POP : int
    :return: NIR as float
    '''
    try:
        max_P = max(list(P.values()))
        length = POP
        return max_P / length
    except Exception:
        return "None"


def hamming_calc(TP, POP):
    '''
    This function calculate hamming_loss
    :param TP: true positive
    :type TP : dict
    :param POP: population
    :type POP : int
    :return: hamming loss as float
    '''
    try:
        length = POP
        return (1 / length) * (length - sum(TP.values()))
    except Exception:
        return "None"


def zero_one_loss_calc(TP, POP):
    '''
    This function zero_one_loss
    :param TP: true Positive
    :type TP : dict
    :param POP: population
    :type POP : int
    :return: zero_one loss as integer
    '''
    try:
        length = POP
        return (length - sum(TP.values()))
    except Exception:
        return "None"


def vector_filter(actual_vector, predict_vector):
    '''
    This function convert different type of items in vectors to str
    :param actual_vector: actual values
    :type actual_vector : list
    :param predict_vector: predict value
    :type predict_vector : list
    :return: new actual and predict vector
    '''
    temp = []
    temp.extend(actual_vector)
    temp.extend(predict_vector)
    types = set(map(type, temp))
    if len(types) > 1:
        return [list(map(str, actual_vector)), list(map(str, predict_vector))]
    return [actual_vector, predict_vector]


def vector_check(vector):
    '''
    This function check input vector items type
    :param vector: input vector
    :type vector : list
    :return: bool
    '''
    for i in vector:
        if isinstance(i, int) is False:
            return False
        if i < 0:
            return False
    return True


def class_check(vector):
    '''
    This function check different items in matrix classes
    :param vector: input vector
    :type vector : list
    :return: bool
    '''
    for i in vector:
        if not isinstance(i, type(vector[0])):
            return False
    return True


def matrix_check(table):
    '''
    This function check input matrix format
    :param table: input matrix
    :type table : dict
    :return: bool
    '''
    try:
        if len(table.keys()) == 0:
            return False
        for i in table.keys():
            if table.keys() != table[i].keys() or vector_check(
                    list(table[i].values())) is False:
                return False
        return True
    except Exception:
        return False


def entropy_calc(item, POP):
    '''
    This function calculate reference and response likelihood
    :param item : TOP or P
    :type item : dict
    :param POP: population
    :type POP : dict
    :return: reference or response likelihood as float
    '''
    try:
        result = 0
        for i in item.keys():
            likelihood = item[i] / POP[i]
            if likelihood != 0:
                result += likelihood * math.log(likelihood, 2)
        return -result
    except Exception:
        return "None"


def kappa_no_prevalence_calc(overall_accuracy):
    '''
    This function calculate kappa no prevalence
    :param overall_accuracy: overall accuracy
    :type overall_accuracy : float
    :return: kappa no prevalence as float
    '''
    try:
        result = 2 * overall_accuracy - 1
        return result
    except Exception:
        return "None"


def cross_entropy_calc(TOP, P, POP):
    '''
    This function calculate cross entropy
    :param TOP: test outcome positive
    :type TOP : dict
    :param P: condition positive
    :type P : dict
    :param POP: population
    :type POP : dict
    :return: cross entropy as float
    '''
    try:
        result = 0
        for i in TOP.keys():
            reference_likelihood = P[i] / POP[i]
            response_likelihood = TOP[i] / POP[i]
            if response_likelihood != 0 and reference_likelihood != 0:
                result += reference_likelihood * \
                    math.log(response_likelihood, 2)
        return -result
    except Exception:
        return "None"


def joint_entropy_calc(classes, table, POP):
    '''
    This function calculate joint entropy
    :param classes: confusion matrix classes
    :type classes : list
    :param table: confusion matrix table
    :type table : dict
    :param POP: population
    :type POP : dict
    :return: joint entropy as float
    '''
    try:
        result = 0
        for i in classes:
            for index, j in enumerate(classes):
                p_prime = table[i][j] / POP[i]
                if p_prime != 0:
                    result += p_prime * math.log(p_prime, 2)
        return -result
    except Exception:
        return "None"


def conditional_entropy_calc(classes, table, P, POP):
    '''
    This function calculate conditional entropy
    :param classes: confusion matrix classes
    :type classes : list
    :param table: confusion matrix table
    :type table : dict
    :param P: condition positive
    :type P : dict
    :param POP: population
    :type POP : dict
    :return: conditional entropy as float
    '''
    try:
        result = 0
        for i in classes:
            temp = 0
            for index, j in enumerate(classes):
                p_prime = 0
                if P[i] != 0:
                    p_prime = table[i][j] / P[i]
                if p_prime != 0:
                    temp += p_prime * math.log(p_prime, 2)
            result += temp * (P[i] / POP[i])
        return -result
    except Exception:
        return "None"


def mutual_information_calc(response_entropy, conditional_entropy):
    '''
    This function calculate mutual information
    :param response_entropy:  response entropy
    :type response_entropy : float
    :param conditional_entropy:  conditional entropy
    :type conditional_entropy : float
    :return: mutual information as float
    '''
    try:
        return response_entropy - conditional_entropy
    except Exception:
        return "None"


def kl_divergence_calc(P, TOP, POP):
    '''
    This function calculate Kullback-Liebler (KL) divergence
    :param P: condition positive
    :type P : dict
    :param TOP: test outcome positive
    :type TOP : dict
    :param POP: population
    :type POP : dict
    :return: Kullback-Liebler (KL) divergence as float
    '''
    try:
        result = 0
        for i in TOP.keys():
            reference_likelihood = P[i] / POP[i]
            response_likelihood = TOP[i] / POP[i]
            result += reference_likelihood * \
                math.log((reference_likelihood / response_likelihood), 2)
        return result
    except Exception:
        return "None"


def lambda_B_calc(classes, table, TOP, POP):
    '''
    This function calculate  Goodman and Kruskal's lambda B
    :param classes: confusion matrix classes
    :type classes : list
    :param table: confusion matrix table
    :type table : dict
    :param TOP: test outcome positive
    :type TOP : dict
    :param POP: population
    :type POP : int
    :return: Goodman and Kruskal's lambda B as float
    '''
    try:
        result = 0
        length = POP
        maxresponse = max(list(TOP.values()))
        for i in classes:
            result += max(list(table[i].values()))
        result = (result - maxresponse) / (length - maxresponse)
        return result
    except Exception:
        return "None"


def lambda_A_calc(classes, table, P, POP):
    '''
    This function calculate Goodman and Kruskal's lambda A
    :param classes: confusion matrix classes
    :type classes : list
    :param table: confusion matrix table
    :type table : dict
    :param P: condition positive
    :type P : dict
    :param POP: population
    :type POP : int
    :return: Goodman and Kruskal's lambda A as float
    '''
    try:
        result = 0
        maxreference = max(list(P.values()))
        length = POP
        for i in classes:
            col = []
            for col_item in table.values():
                col.append(col_item[i])
            result += max(col)
        result = (result - maxreference) / (length - maxreference)
        return result
    except Exception:
        return "None"


def chi_square_calc(classes, table, TOP, P, POP):
    '''
    This function calculate chi-squared
    :param classes: confusion matrix classes
    :type classes : list
    :param table: confusion matrix table
    :type table : dict
    :param TOP: test outcome positive
    :type TOP : dict
    :param P: condition positive
    :type P : dict
    :param POP: population
    :type POP : dict
    :return: chi-squared as float
    '''
    try:
        result = 0
        for i in classes:
            for index, j in enumerate(classes):
                expected = (TOP[j] * P[i]) / (POP[i])
                result += ((table[i][j] - expected)**2) / expected
        return result
    except Exception:
        return "None"


def phi_square_calc(chi_square, POP):
    '''
    This function calculate phi_squared
    :param chi_square: chi squared
    :type chi_square : float
    :param POP: population
    :type POP : int
    :return: phi_squared as float
    '''
    try:
        return chi_square / POP
    except Exception:
        return "None"


def cramers_V_calc(phi_square, classes):
    '''
    This function calculate Cramer's V
    :param phi_square: phi_squared
    :type phi_square : float
    :param classes: confusion matrix classes
    :type classes : list
    :return: Cramer's V as float
    '''
    try:
        return math.sqrt((phi_square / (len(classes) - 1)))
    except Exception:
        return "None"


def DF_calc(classes):
    '''
    This function calculate chi squared degree of freedom
    :param classes: confusion matrix classes
    :type classes : list
    :return: DF as int
    '''
    try:
        return (len(classes) - 1)**2
    except Exception:
        return "None"


def TTPN_calc(item1, item2):
    '''
    This function calculate TPR,TNR,PPV,NPV
    :param item1: item1 in fractional expression
    :type item1 : int
    :param item2: item2 in fractional expression
    :type item2: int
    :return: result as float
    '''
    try:
        result = item1 / (item1 + item2)
        return result
    except ZeroDivisionError:
        return "None"


def FXR_calc(item):
    '''
    This function calculate FNR,FPR,FDR,FOR
    :param item: item In expression
    :type item:float
    :return: result as float
    '''
    try:
        result = 1 - item
        return result
    except Exception:
        return "None"


def ACC_calc(TP, TN, FP, FN):
    '''
    This function calculate accuracy
    :param TP: true positive
    :type TP : int
    :param TN: true negative
    :type TN : int
    :param FP: false positive
    :type FP : int
    :param FN: false negative
    :type FN : int
    :return: accuracy as float
    '''
    try:
        result = (TP + TN) / (TP + TN + FN + FP)
        return result
    except ZeroDivisionError:
        return "None"


def ERR_calc(ACC):
    '''
    This function calculate error rate
    :param ACC: accuracy
    :type ACC: float
    :return: error rate as float
    '''
    try:
        return 1 - ACC
    except Exception:
        return "None"


def F_calc(TP, FP, FN, Beta):
    '''
    This function calculate F score
    :param TP: true positive
    :type TP : int
    :param FP: false positive
    :type FP : int
    :param FN: false negative
    :type FN : int
    :param Beta : beta coefficient
    :type Beta : float
    :return: F score as float
    '''
    try:
        result = ((1 + (Beta)**2) * TP) / \
            ((1 + (Beta)**2) * TP + FP + (Beta**2) * FN)
        return result
    except ZeroDivisionError:
        return "None"


def MCC_calc(TP, TN, FP, FN):
    '''
    This function calculate MCC (Matthews correlation coefficient)
    :param TP: true positive
    :type TP : int
    :param TN: true negative
    :type TN : int
    :param FP: false positive
    :type FP : int
    :param FN: false negative
    :type FN : int
    :return: MCC as float
    '''
    try:
        result = (TP * TN - FP * FN) / \
            (math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
        return result
    except ZeroDivisionError:
        return "None"


def MK_BM_calc(item1, item2):
    '''
    This function calculate BM (Informedness) and MK (Markedness)
    :param item1: item1 in expression
    :type item1:float
    :param item2: item2 in expression
    :type item2:float
    :return: MK and BM as float
    '''
    try:
        result = item1 + item2 - 1
        return result
    except Exception:
        return "None"


def LR_calc(item1, item2):
    '''
    This function calculate likelihood ratio
    :param item1: item1 in expression
    :type item1:float
    :param item2: item2 in expression
    :type item2:float
    :return: LR+ and LR- as float
    '''
    try:
        result = item1 / item2
        return result
    except Exception:
        return "None"


def PRE_calc(P, POP):
    '''
    This function calculate prevalence
    :param P: condition positive
    :type P : int
    :param POP: population
    :type POP : int
    :return: prevalence as float
    '''
    try:
        result = P / POP
        return result
    except Exception:
        return "None"


def G_calc(PPV, TPR):
    '''
    This function calculate G-measure
    :param PPV:  precision or positive predictive value
    :type PPV : float
    :param TPR: sensitivity, recall, hit rate, or true positive rate
    :type TPR : float
    :return: G-measure as float
    '''
    try:
        result = math.sqrt(PPV * TPR)
        return result
    except Exception:
        return "None"


def RACCU_calc(TOP, P, POP):
    '''
    This function calculate RACCU (Random accuracy unbiased)
    :param TOP: test outcome positive
    :type TOP : int
    :param P: condition positive
    :type P : int
    :param POP: population
    :type POP : int
    :return: RACCU as float
    '''
    result = ((TOP + P) / (2 * POP))**2
    return result


def RACC_calc(TOP, P, POP):
    '''
    This function calculate random accuracy
    :param TOP: test outcome positive
    :type TOP : int
    :param P:  condition positive
    :type P : int
    :param POP: population
    :type POP:int
    :return: RACC as float
    '''
    result = (TOP * P) / ((POP) ** 2)
    return result


def reliability_calc(RACC, ACC):
    '''
    This function calculate reliability
    :param RACC: random accuracy
    :type RACC : float
    :param ACC: accuracy
    :type ACC : float
    :return: reliability as float
    '''
    try:
        result = (ACC - RACC) / (1 - RACC)
        return result
    except Exception:
        return "None"


def kappa_se_calc(PA, PE, POP):
    '''
    This function calculate kappa standard error
    :param PA: observed agreement among raters (overall accuracy)
    :type PA : float
    :param PE:  hypothetical probability of chance agreement (random accuracy)
    :type PE : float
    :param POP: population
    :type POP:int
    :return: kappa standard error as float
    '''
    try:
        result = math.sqrt((PA * (1 - PA)) / (POP * ((1 - PE)**2)))
        return result
    except Exception:
        return "None"


def CI_calc(mean, SE, CV=1.96):
    '''
    This function calculate confidence interval
    :param mean: mean of data
    :type mean : float
    :param SE: standard error of data
    :type SE : float
    :param CV: critical value
    :type CV:float
    :return: confidence interval as tuple
    '''
    try:
        CI_down = mean - CV * SE
        CI_up = mean + CV * SE
        return (CI_down, CI_up)
    except Exception:
        return ("None", "None")


def se_calc(overall_accuracy, POP):
    '''
    This function calculate standard error with binomial distribution
    :param overall_accuracy: overall accuracy
    :type  overall_accuracy : float
    :type PE : float
    :param POP: population
    :type POP : int
    :return: standard error as float
    '''
    try:
        return math.sqrt(
            (overall_accuracy * (1 - overall_accuracy)) / POP)
    except Exception:
        return "None"


def micro_calc(TP, item):
    '''
    This function calculate PPV_Micro and TPR_Micro
    :param TP: true positive
    :type TP:dict
    :param item: FN or FP
    :type item : dict
    :return: PPV_Micro or TPR_Micro as float
    '''
    try:
        TP_sum = sum(TP.values())
        item_sum = sum(item.values())
        return TP_sum / (TP_sum + item_sum)
    except Exception:
        return "None"


def macro_calc(item):
    '''
    This function calculate PPV_Macro and TPR_Macro
    :param item: PPV or TPR
    :type item:dict
    :return: PPV_Macro or TPR_Macro as float
    '''
    try:
        item_sum = sum(item.values())
        item_len = len(item.values())
        return item_sum / item_len
    except Exception:
        return "None"


def PC_PI_calc(P, TOP, POP):
    '''
    This function calculate percent chance agreement for Scott's Pi
    :param P: condition positive
    :type P : dict
    :param TOP: test outcome positive
    :type TOP : dict
    :param POP: population
    :type POP:dict
    :return: percent chance agreement as float
    '''
    try:
        result = 0
        for i in P.keys():
            result += ((P[i] + TOP[i]) / (2 * POP[i]))**2
        return result
    except Exception:
        return "None"


def PC_AC1_calc(P, TOP, POP):
    '''
    This function calculate percent chance agreement for Gwet's AC1
    :param P: condition positive
    :type P : dict
    :param TOP: test outcome positive
    :type TOP : dict
    :param POP: population
    :type POP:dict
    :return: percent chance agreement as float
    '''
    try:
        result = 0
        classes = list(P.keys())
        for i in classes:
            pi = ((P[i] + TOP[i]) / (2 * POP[i]))
            result += pi * (1 - pi)
        result = result / (len(classes) - 1)
        return result
    except Exception:
        return "None"


def PC_S_calc(classes):
    '''
    This function calculate percent chance agreement for Bennett-et-al.'s-S-score
    :param classes: confusion matrix classes
    :type classes : list
    :return: percent chance agreement as float
    '''
    try:
        return 1 / (len(classes))
    except Exception:
        return "None"


def jaccard_index_calc(TP, TOP, P):
    '''
    This function calculate Jaccard index for each class
    :param TP: true positive
    :type TP : int
    :param TOP: test outcome positive
    :type TOP : int
    :param P:  condition positive
    :type P : int
    :return: Jaccard index as float
    '''
    try:
        return TP / (TOP + P - TP)
    except Exception:
        return "None"


def overall_jaccard_index_calc(jaccard_list):
    '''
    This function calculate overall jaccard index
    :param jaccard_list : list of jaccard index for each class
    :type jaccard_list : list
    :return: (jaccard_sum , jaccard_mean) as tuple
    '''
    try:
        jaccard_sum = sum(jaccard_list)
        jaccard_mean = jaccard_sum / len(jaccard_list)
        return (jaccard_sum, jaccard_mean)
    except Exception:
        return "None"


def overall_accuracy_calc(TP, POP):
    '''
    This function calculate overall accuracy
    :param TP: true positive
    :type TP : dict
    :param POP: population
    :type POP:int
    :return: overall_accuracy as float
    '''
    try:
        overall_accuracy = sum(TP.values()) / POP
        return overall_accuracy
    except Exception:
        return None


def overall_random_accuracy_calc(item):
    '''
    This function calculate overall random accuracy
    :param item: RACC or RACCU
    :type item : dict
    :return: overall random accuracy as float
    '''
    try:
        return sum(item.values())
    except Exception:
        return "None"


def overall_statistics(
        RACC,
        RACCU,
        TPR,
        PPV,
        TP,
        FN,
        FP,
        POP,
        P,
        TOP,
        jaccard_list,
        CEN_dict,
        MCEN_dict,
        AUC_dict,
        classes,
        table):
    '''
    This function return overall statistics
    :param RACC: random accuracy
    :type RACC : dict
    :param TPR: sensitivity, recall, hit rate, or true positive rate
    :type TPR : dict
    :param PPV: precision or positive predictive value
    :type PPV : dict
    :param TP: true positive
    :type TP : dict
    :param FN: false negative
    :type FN : dict
    :param FP: false positive
    :type FP: dict
    :param POP: population
    :type POP:dict
    :param P: condition positive
    :type P : dict
    :param POP: population
    :type POP:dict
    :param TOP: test outcome positive
    :type TOP : dict
    :param jaccard_list : list of jaccard index for each class
    :type jaccard_list : list
    :param CEN_dict: CEN dictionary for each class
    :type CEN_dict : dict
    :param classes: confusion matrix classes
    :type classes : list
    :param table: input matrix
    :type table : dict
    :return: overall statistics as dict
    '''
    population = list(POP.values())[0]
    overall_accuracy = overall_accuracy_calc(TP, population)
    overall_random_accuracy_unbiased = overall_random_accuracy_calc(RACCU)
    overall_random_accuracy = overall_random_accuracy_calc(RACC)
    overall_kappa = reliability_calc(overall_random_accuracy, overall_accuracy)
    PC_PI = PC_PI_calc(P, TOP, POP)
    PC_AC1 = PC_AC1_calc(P, TOP, POP)
    PC_S = PC_S_calc(classes)
    PI = reliability_calc(PC_PI, overall_accuracy)
    AC1 = reliability_calc(PC_AC1, overall_accuracy)
    S = reliability_calc(PC_S, overall_accuracy)
    kappa_SE = kappa_se_calc(
        overall_accuracy,
        overall_random_accuracy, population)
    kappa_unbiased = reliability_calc(
        overall_random_accuracy_unbiased,
        overall_accuracy)
    kappa_no_prevalence = kappa_no_prevalence_calc(overall_accuracy)
    kappa_CI = CI_calc(overall_kappa, kappa_SE)
    overall_accuracy_se = se_calc(overall_accuracy, population)
    overall_accuracy_CI = CI_calc(overall_accuracy, overall_accuracy_se)
    chi_squared = chi_square_calc(classes, table, TOP, P, POP)
    phi_squared = phi_square_calc(chi_squared, population)
    cramer_V = cramers_V_calc(phi_squared, classes)
    response_entropy = entropy_calc(TOP, POP)
    reference_entropy = entropy_calc(P, POP)
    cross_entropy = cross_entropy_calc(TOP, P, POP)
    join_entropy = joint_entropy_calc(classes, table, POP)
    conditional_entropy = conditional_entropy_calc(classes, table, P, POP)
    mutual_information = mutual_information_calc(
        response_entropy, conditional_entropy)
    kl_divergence = kl_divergence_calc(P, TOP, POP)
    lambda_B = lambda_B_calc(classes, table, TOP, population)
    lambda_A = lambda_A_calc(classes, table, P, population)
    DF = DF_calc(classes)
    overall_jaccard_index = overall_jaccard_index_calc(list(
        jaccard_list.values()))
    hamming_loss = hamming_calc(TP, population)
    zero_one_loss = zero_one_loss_calc(TP, population)
    NIR = NIR_calc(P, population)
    p_value = p_value_calc(TP, population, NIR)
    overall_CEN = overall_CEN_calc(classes, TP, TOP, P, CEN_dict)
    overall_MCEN = overall_CEN_calc(classes, TP, TOP, P, MCEN_dict, True)
    overall_MCC = overall_MCC_calc(classes, table, TOP, P)
    RR = RR_calc(classes, TOP)
    CBA = CBA_calc(classes, table, TOP, P)
    AUNU = macro_calc(AUC_dict)
    AUNP = AUNP_calc(classes, P, POP, AUC_dict)
    RCI = RCI_calc(mutual_information, reference_entropy)
    return {
        "Overall ACC": overall_accuracy,
        "Kappa": overall_kappa,
        "Overall RACC": overall_random_accuracy,
        "SOA1(Landis & Koch)": kappa_analysis_koch(overall_kappa),
        "SOA2(Fleiss)": kappa_analysis_fleiss(overall_kappa),
        "SOA3(Altman)": kappa_analysis_altman(overall_kappa),
        "SOA4(Cicchetti)": kappa_analysis_cicchetti(overall_kappa),
        "TPR Macro": macro_calc(TPR),
        "PPV Macro": macro_calc(PPV),
        "TPR Micro": micro_calc(
            TP=TP,
            item=FN),
        "PPV Micro": micro_calc(
            TP=TP,
            item=FP),
        "Scott PI": PI,
        "Gwet AC1": AC1,
        "Bennett S": S,
        "Kappa Standard Error": kappa_SE,
        "Kappa 95% CI": kappa_CI,
        "Chi-Squared": chi_squared,
        "Phi-Squared": phi_squared,
        "Cramer V": cramer_V,
        "Chi-Squared DF": DF,
        "95% CI": overall_accuracy_CI,
        "Standard Error": overall_accuracy_se,
        "Response Entropy": response_entropy,
        "Reference Entropy": reference_entropy,
        "Cross Entropy": cross_entropy,
        "Joint Entropy": join_entropy,
        "Conditional Entropy": conditional_entropy,
        "KL Divergence": kl_divergence,
        "Lambda B": lambda_B,
        "Lambda A": lambda_A,
        "Kappa Unbiased": kappa_unbiased,
        "Overall RACCU": overall_random_accuracy_unbiased,
        "Kappa No Prevalence": kappa_no_prevalence,
        "Mutual Information": mutual_information,
        "Overall J": overall_jaccard_index,
        "Hamming Loss": hamming_loss,
        "Zero-one Loss": zero_one_loss,
        "NIR": NIR,
        "P-Value": p_value,
        "Overall CEN": overall_CEN,
        "Overall MCEN": overall_MCEN,
        "Overall MCC": overall_MCC,
        "RR": RR,
        "CBA": CBA,
        "AUNU": AUNU,
        "AUNP": AUNP,
        "RCI": RCI}


def class_statistics(TP, TN, FP, FN, classes, table):
    '''
    This function return all class statistics
    :param TP: true positive dict for all classes
    :type TP : dict
    :param TN: true negative dict for all classes
    :type TN : dict
    :param FP: false positive dict for all classes
    :type FP : dict
    :param FN: false negative dict for all classes
    :type FN : dict
    :param classes: classes
    :type classes : list
    :param table: input matrix
    :type table : dict
    :return: result as dict
    '''
    TPR = {}
    TNR = {}
    PPV = {}
    NPV = {}
    FNR = {}
    FPR = {}
    FDR = {}
    FOR = {}
    ACC = {}
    F1_SCORE = {}
    MCC = {}
    BM = {}
    MK = {}
    PLR = {}
    NLR = {}
    DOR = {}
    POP = {}
    P = {}
    N = {}
    TOP = {}
    TON = {}
    PRE = {}
    G = {}
    RACC = {}
    F05_Score = {}
    F2_Score = {}
    ERR = {}
    RACCU = {}
    Jaccrd_Index = {}
    IS = {}
    CEN = {}
    MCEN = {}
    AUC = {}
    dInd = {}
    sInd = {}
    DP = {}
    Y = {}
    PLRI = {}
    DPI = {}
    AUCI = {}
    GI = {}
    LS = {}
    AM = {}
    BCD = {}
    for i in TP.keys():
        POP[i] = TP[i] + TN[i] + FP[i] + FN[i]
        P[i] = TP[i] + FN[i]
        N[i] = TN[i] + FP[i]
        TOP[i] = TP[i] + FP[i]
        TON[i] = TN[i] + FN[i]
        TPR[i] = TTPN_calc(TP[i], FN[i])
        TNR[i] = TTPN_calc(TN[i], FP[i])
        PPV[i] = TTPN_calc(TP[i], FP[i])
        NPV[i] = TTPN_calc(TN[i], FN[i])
        FNR[i] = FXR_calc(TPR[i])
        FPR[i] = FXR_calc(TNR[i])
        FDR[i] = FXR_calc(PPV[i])
        FOR[i] = FXR_calc(NPV[i])
        ACC[i] = ACC_calc(TP[i], TN[i], FP[i], FN[i])
        F1_SCORE[i] = F_calc(TP[i], FP[i], FN[i], 1)
        F05_Score[i] = F_calc(TP[i], FP[i], FN[i], 0.5)
        F2_Score[i] = F_calc(TP[i], FP[i], FN[i], 2)
        MCC[i] = MCC_calc(TP[i], TN[i], FP[i], FN[i])
        BM[i] = MK_BM_calc(TPR[i], TNR[i])
        MK[i] = MK_BM_calc(PPV[i], NPV[i])
        PLR[i] = LR_calc(TPR[i], FPR[i])
        NLR[i] = LR_calc(FNR[i], TNR[i])
        DOR[i] = LR_calc(PLR[i], NLR[i])
        PRE[i] = PRE_calc(P[i], POP[i])
        G[i] = G_calc(PPV[i], TPR[i])
        RACC[i] = RACC_calc(TOP[i], P[i], POP[i])
        ERR[i] = ERR_calc(ACC[i])
        RACCU[i] = RACCU_calc(TOP[i], P[i], POP[i])
        Jaccrd_Index[i] = jaccard_index_calc(TP[i], TOP[i], P[i])
        IS[i] = IS_calc(TP[i], FP[i], FN[i], POP[i])
        CEN[i] = CEN_calc(classes, table, TOP[i], P[i], i)
        MCEN[i] = CEN_calc(classes, table, TOP[i], P[i], i, True)
        AUC[i] = AUC_calc(TNR[i], TPR[i])
        dInd[i] = dInd_calc(TNR[i], TPR[i])
        sInd[i] = sInd_calc(dInd[i])
        DP[i] = DP_calc(TPR[i], TNR[i])
        Y[i] = BM[i]
        PLRI[i] = PLR_analysis(PLR[i])
        DPI[i] = DP_analysis(DP[i])
        AUCI[i] = AUC_analysis(AUC[i])
        GI[i] = GI_calc(AUC[i])
        LS[i] = lift_calc(PPV[i], PRE[i])
        AM[i] = AM_calc(TOP[i], P[i])
    for i in TP.keys():
        BCD[i] = BCD_calc(TOP, P, AM[i])
    result = {
        "TPR": TPR,
        "TNR": TNR,
        "PPV": PPV,
        "NPV": NPV,
        "FNR": FNR,
        "FPR": FPR,
        "FDR": FDR,
        "FOR": FOR,
        "ACC": ACC,
        "F1": F1_SCORE,
        "MCC": MCC,
        "BM": BM,
        "MK": MK,
        "PLR": PLR,
        "NLR": NLR,
        "DOR": DOR,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "POP": POP,
        "P": P,
        "N": N,
        "TOP": TOP,
        "TON": TON,
        "PRE": PRE,
        "G": G,
        "RACC": RACC,
        "F0.5": F05_Score,
        "F2": F2_Score,
        "ERR": ERR,
        "RACCU": RACCU,
        "J": Jaccrd_Index,
        "IS": IS,
        "CEN": CEN,
        "MCEN": MCEN,
        "AUC": AUC,
        "sInd": sInd,
        "dInd": dInd,
        "DP": DP,
        "Y": Y,
        "PLRI": PLRI,
        "DPI": DPI,
        "AUCI": AUCI,
        "GI": GI,
        "LS": LS,
        "AM": AM,
        "BCD": BCD}
    return result
