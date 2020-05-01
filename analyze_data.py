from glob import glob
import numpy as np
from os import path
import pandas as pd
import pickle


def load_data(data_dirs):
    dfs = []
    for data_dir in data_dirs:
        info_filenames = glob(path.join(data_dir, "*_info.pickle"))
        df = pd.DataFrame()
        for info_filename in info_filenames:
            with open(info_filename, "rb") as info_file:
                record = pickle.load(info_file)
                record["data_file_name"] = info_filename[:-12] + "_data.pickle"
                df = df.append(record, ignore_index=True)
        dfs.append(df)
    return pd.concat(dfs)

def xeb_agg_fn(df):
    return pd.Series({
        "n_samples": sum(df["n_samples"]),
        "xeb_mean": np.average(df["xeb_mean"], weights=df["n_samples"]),
        "xeb_std": np.sqrt(
            (
                np.average((df["xeb_std"] ** 2 * df["n_samples"] + df["xeb_mean"] ** 2), weights=df["n_samples"]) -
                np.average(df["xeb_mean"], weights=df["n_samples"]) ** 2
            ) / sum(df["n_samples"])
        )
    })

def rcb_agg_fn(df):
    return pd.Series({
        "n_samples": sum(df["n_samples"]),
        "deviation": np.average(df["deviation"], weights=df["n_samples"]),
        "error": np.average(df["error"] * np.sqrt(df["n_samples"]), weights=df["n_samples"]) / np.sqrt(sum(df["n_samples"]))
    })

def analyze_xeb_benchmark(df, design_settings_columns):
    sorted_df = df.sort_values(["design"] + design_settings_columns + ["moment"])[["design"] + design_settings_columns + ["moment", "n_samples", "xeb_mean", "xeb_std"]]
    grouped = sorted_df.groupby(["design"] + design_settings_columns + ["moment"]).apply(xeb_agg_fn).reset_index()
    grouped["rel_xeb_std"] = grouped["xeb_std"] / grouped["xeb_mean"]
    return grouped

def analyze_rcb_benchmark(df, design_settings_columns):
    sorted_df = df.sort_values(["design"] + design_settings_columns + ["n_tensor_factors"])[["design"] + design_settings_columns + ["n_tensor_factors", "deviation", "error", "n_samples"]]
    #grouped = sorted_df.groupby(["design"] + design_settings_columns + ["n_tensor_factors"]).apply(rcb_agg_fn).reset_index()
    #grouped["rel_error"] = grouped["error"] / grouped["deviation"]
    grouped = sorted_df
    return grouped

def generate_xeb_statistics(df, design_settings_columns, statistics, d_from_design_settings):
    def statistics_generator(partial_df):
        d_value = d_from_design_settings(partial_df.iloc[0])
        samples = []
        for data_file_name in partial_df["data_file_name"]:
            with open(data_file_name, "rb") as data_file:
                record = pickle.load(data_file)
            samples.extend(record["samples"])
        statistics_dict = {}
        for statistic in statistics:
            if statistic == "log":
                statistics_dict["log"] = (np.log(d_value) - np.mean(-np.log(samples))) / (1 - np.euler_gamma)
                statistics_dict["log_std"] = np.std(-np.log2(samples)) / (np.euler_gamma * sum(partial_df["n_samples"]))
            else:
                statistics_dict[f"moment_{statistic}"] = 1 + (np.mean((d_value * np.array(samples)) ** statistic) - 1 - (np.math.factorial(statistic + 1) - 1)) / (np.math.factorial(statistic + 1) - 1)
                statistics_dict[f"moment_{statistic}_std"] = np.std((d_value * np.array(samples)) ** statistic) / ((np.math.factorial(statistic + 1) - 1) * np.sqrt(sum(partial_df["n_samples"])))
            statistics_dict["n_samples"] = sum(partial_df["n_samples"])
        return pd.Series(statistics_dict)
    sorted_df = df.sort_values(["design"] + design_settings_columns)
    return sorted_df.groupby(["design"] + design_settings_columns).apply(statistics_generator)#.reset_index()