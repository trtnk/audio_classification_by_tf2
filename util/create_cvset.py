from util.common import *

import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def get_cv_df_from_yaml(config_yaml_path, output_csv_path=None):
    config = yaml_load(config_yaml_path)

    directory_label_name = config["directory_label_name"] if "directory_label_name" in config else []
    file_label_name = config["file_label_name"] if "file_label_name" in config else []

    # create file information df
    info_df = get_file_info_df(config["data_base_directory"],
                               config["data_extension"],
                               label_csv_path=config["label_csv_path"],
                               directory_label_name=directory_label_name,
                               file_label_name=file_label_name)

    # filter
    if "data_constraint" in config:
        info_df = df_extractor(info_df, config["data_constraint"])

    # main label setting
    if "default_main_label" in config["main_label"]:
        main_label = config["main_label"]
    elif "create_main_label" in config["main_label"]:
        info_df = add_new_column(info_df, "_main_label_", config["main_label"]["create_main_label"], dropna=True)
        main_label = "_main_label_"

    if "test_id_path" in config:
        test_id_list = list(pd.read_csv(config["test_id_path"], header=None)[0].values)
    else:
        test_id_list = None

    if "multi_file_input" in config:
        multi_file_input = True
        multi_file_input_config = config["multi_file_input"]
    else:
        multi_file_input = False
        multi_file_input_config = {}

    random_state = config["cv"]["random_state"] if "random_state" in config["cv"] else None
    grouped_label = config["cv"]["grouped_label"] if "grouped_label" in config["cv"] else None

    # create cvset df
    cv_df = get_cv_df(info_df, main_label, n_splits=config["cv"]["n_splits"], shuffle=bool(config["cv"]["shuffle"]),
                      test_id_list=test_id_list, id_column=config["id_column"],
                      grouped_label=grouped_label,
                      balanced=config["cv"]["balanced"], balanced_other_label=config["cv"]["balanced_other_label"],
                      stratified=config["cv"]["stratified"], stratified_other_labels=config["cv"]["stratified_other_labels"],
                      multi_file_input=multi_file_input,
                      multi_file_identification_info=multi_file_input_config,
                      random_state=random_state,
                      output_csv_path=output_csv_path)
    return cv_df


def get_cv_df(info_df, main_label,
              n_splits=5, shuffle=True,
              test_id_list=None, id_column=None,
              grouped_label=None,
              balanced=True, balanced_other_label=None,
              stratified=False, stratified_other_labels=None,
              multi_file_input=False,
              multi_file_identification_info={"base_column": "", "divide_column": "", "order":[]},
              random_state=None,
              output_csv_path=None):

    cv_df = info_df.copy()
    cv_df.loc[:, "cvset"] = pd.NA

    info_df_ = info_df.copy()

    if test_id_list is not None:
        test_idx = info_df_[info_df_[id_column].isin(test_id_list)].index
        cv_df.loc[test_idx, "cvset"] = "test"
        info_df_ = info_df_[~info_df_[id_column].isin(test_id_list)]

    # grouped
    if grouped_label is not None:
        key_df = info_df_.drop_duplicates(subset=grouped_label)
    else:
        key_df = info_df_.copy()

    # balanced
    if balanced:
        label_vals = list(dict.fromkeys(key_df[main_label].values))
        min_sample_num = key_df[main_label].value_counts().min()
        key_df_ = pd.DataFrame({})
        for label_val in label_vals:
            each_label_key_df = key_df[key_df[main_label] == label_val]
            if balanced_other_label is None:
                key_df_ = key_df_.append(each_label_key_df.sample(min_sample_num, random_state=random_state)).reset_index(drop=True)
            else:
                key_df_ = key_df_.append(sample_balanced(each_label_key_df, min_sample_num, balanced_other_label, random_state=random_state)).reset_index(drop=True)
    else:
        key_df_ = key_df

    # stratified
    if stratified:
        if stratified_other_labels is not None:
            kf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state).split(key_df_, key_df_[[main_label]+stratified_other_labels])
        else:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state).split(key_df_, key_df_[main_label])
    else:
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state).split(key_df_, key_df_[main_label])

    # create cv set
    for cvset, (_, eval_idx) in enumerate(kf, start=1):
        if grouped_label is None:
            grouped_label = id_column
        eval_keys = key_df_.loc[eval_idx, :][grouped_label]
        cv_df.loc[cv_df[grouped_label].isin(eval_keys), "cvset"] = f"cv{cvset}"

    cv_df = cv_df.dropna(subset=["cvset"])

    if multi_file_input:
        base_column = multi_file_identification_info["base_column"]
        divide_column = multi_file_identification_info["divide_column"]
        order = multi_file_identification_info["order"]

        cv_df_ = cv_df.copy()
        cv_df_ = cv_df_.drop_duplicates(subset=base_column)

        for id_, new_df in cv_df.groupby(base_column):
            file_paths = [new_df[new_df[divide_column] == val]["filepath"].values[0] for val in order]
            idx = cv_df_.loc[cv_df_[base_column] == id_, "filepath"].index[0]
            cv_df_.at[idx, "filepath"] = file_paths
        cv_df = cv_df_.reset_index(drop=True)

    if output_csv_path is not None:
        cv_df.to_csv(output_csv_path, index=False)

    return cv_df
