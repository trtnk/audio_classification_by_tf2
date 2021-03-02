from util.create_cvset import get_cv_df_from_yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-yp", "--yaml_path", help="config yaml file path")
parser.add_argument("-op", "--output_path", default=None, help='output csv file path')

args = parser.parse_args()

get_cv_df_from_yaml(args.yaml_path, output_csv_path=args.output_path)
