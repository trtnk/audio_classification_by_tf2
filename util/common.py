import os
import glob
import yaml
import codecs
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score


def yaml_load(file_path):
    with codecs.open(file_path, "r", "utf-8") as stream:
        param = yaml.safe_load(stream)
    return param


def df_extractor(df, fix_option):
    if "operator" in fix_option:
        if not len(fix_option["label"]) == len(fix_option["value"]) == len(fix_option["operator"]):
            raise Exception["fix_option element count of label or value or operator is not save."]
    else:
        if not len(fix_option["label"]) == len(fix_option["value"]):
            raise Exception["fix_option element count of label or value is not save."]
        fix_option["operator"] = ["==" for i in range(len(fix_option["label"]))]

    ret_df = df.copy()

    query_str = ""
    for i in range(len(fix_option["label"])):
        if type(fix_option["value"][i]) == list:
            ret_df = ret_df[ret_df[fix_option["label"][i]].isin(fix_option["value"][i])]
        else:
            query_str += fix_option["label"][i]
            query_str += f" {fix_option['operator'][i]} "
            if type(fix_option["value"][i]) == str:
                val = f"'{fix_option['value'][i]}'"
            else:
                val = str(fix_option["value"][i])
            query_str += val+" and " if i != len(fix_option["label"])-1 else val

    if len(query_str) == 0:
        return ret_df
    else:
        return ret_df.query(query_str)


def divide_filepath(file_path):
    rest, filename = os.path.split(file_path)

    dir_list = []
    while(os.path.split(rest)[1] != ''):
        rest, part_str = os.path.split(rest)
        dir_list.append(part_str)
    return filename, dir_list


def add_new_column(df, label_name, each_fix_option, dropna=True):
    df_ = pd.DataFrame({})
    for label_val in each_fix_option:
        df_tmp = df_extractor(df, each_fix_option[label_val])
        df_tmp.loc[:, label_name] = label_val
        df_ = df_.append(df_tmp)
    if dropna:
        ret_df = df_.reset_index(drop=True)
    else:
        ret_df = df.copy()
        ret_df.loc[:, label_name] = pd.NA
        ret_df.loc[df_.index, label_name] = df_.loc[:, label_name]
        ret_df = ret_df.reset_index(drop=True)
    return ret_df


def get_file_info_df(data_dir, extension, label_csv_path=None, directory_label_name=[], file_label_name=[]):
    files = sorted(glob.glob(os.path.join(os.path.abspath(data_dir), f"**/*.{extension}")))
    filename, dirs = divide_filepath(files[0].replace(os.path.abspath(data_dir), ""))
    file_parts = os.path.splitext(filename)[0].split('_')

    df_column = file_label_name + directory_label_name
    df_values = []
    for file in files:
        val = file.replace(os.path.abspath(data_dir), "")
        filename, dirs = divide_filepath(val)

        filename_parts = os.path.splitext(filename)[0].split('_')
        if len(directory_label_name) > 0:
            df_values.append(filename_parts + dirs)
        else:
            df_values.append(filename_parts)

    info_df = pd.DataFrame(df_values, columns=df_column)
    info_df.loc[:, "filepath"] = files

    if label_csv_path is not None:
        label_df = pd.read_csv(label_csv_path)
        id_column = label_df.columns[0] # csvファイルの１列目の列名が称号用のdata_id
        if id_column in info_df.columns:
            info_df = pd.merge(info_df, label_df, on=id_column)
        else:
            raise Exception(f"The column name for merging label: {id_column} connot be found.")
    return info_df


def sample_balanced(df, sample_num, balanced_label, random_state=None):
    df_ = df.copy()
    class_list = list(dict.fromkeys(df_.loc[:, balanced_label]))

    N_each_class = sample_num // len(class_list)
    N_rest = sample_num % len(class_list)
    ret_df = pd.DataFrame({})
    while (N_each_class > 0):
        pickup_num = 0
        remove_class_list = []
        for class_ in class_list:
            df_tmp = df_[df_[balanced_label] == class_]

            if len(df_tmp) < N_each_class:
                samples = df_tmp
                pickup_num += N_each_class - len(samples)
                remove_class_list.append(class_)
            else:
                samples = df_tmp.sample(N_each_class, random_state=random_state)
            ret_df = ret_df.append(samples)
            df_ = df_.drop(samples.index)

        for remove_class in remove_class_list:
            class_list.remove(remove_class)

        N_each_class = pickup_num // len(class_list)
        N_rest += pickup_num % len(class_list)

        if N_each_class == 0:
            N_each_class = N_rest // len(class_list)
            N_rest = N_rest % len(class_list)
    if N_rest > 0:
        for i in range(N_rest):
            df_tmp = df_[df_[balanced_label] == class_list[i]]
            ret_df = ret_df.append(df_tmp.sample(1, random_state=random_state))
    return ret_df


def sample_balanced_array(arr, label, sample_num, random_state=None):
    arr = np.array(arr)
    label = np.array(label)
    if label.ndim != 1:
        raise Exception("Dimension of label must be 1.")
    if arr.shape[0] != len(label):
        raise Exception("Length of arr and label is different.")
    df = pd.DataFrame(arr)
    df.loc[:, "label"] = label
    ret_df = sample_balanced(df, sample_num, "label", random_state=random_state)
    return ret_df[ret_df.columns[ret_df.columns != "label"]].values, ret_df["label"].values


def evaluate(y_true, y_pred, y_scores, additional_condition=None):
    ret = {}
    ret["accuracy"] = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average=None)
    ret["recall_negative"] = recall[0]
    ret["recall_positive"] = recall[1]
    precision = precision_score(y_true, y_pred, average=None)
    ret["precision_negative"] = precision[0]
    ret["precision_positive"] = precision[1]
    f1 = f1_score(y_true, y_pred, average=None)
    ret["f1_negative"] = f1[0]
    ret["f1_positive"] = f1[1]

    MAX_FALSE_POSITIVE_RATE = 0.3
    ret["AUC"] = roc_auc_score(y_true, y_scores)
    ret["pAUC"] = roc_auc_score(y_true, y_scores, max_fpr=MAX_FALSE_POSITIVE_RATE)

    if additional_condition is not None:
        for key in additional_condition:
            ret[key] = additional_condition[key]
    return pd.Series(ret)
