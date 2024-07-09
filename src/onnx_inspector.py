# SPDX-FileCopyrightText: 2024 (C) 2024 Marco Celoria <celoria.marco@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys
from pathlib import Path
import onnx
from argparser import *


def main(onnx_path, output_dir):
    result_dir    = output_dir + "/results"
    if not(os.path.exists(output_dir) and os.path.isdir(output_dir)):
        os.makedirs(output_dir)
    if not(os.path.exists(result_dir) and os.path.isdir(result_dir)):
        os.makedirs(result_dir)
    print(f"Checking onnx file: {onnx_path}")
    # Check the model
    try:
        onnx.checker.check_model(onnx_path, full_check=True)
    except onnx.checker.ValidationError as e:
        print("The model is invalid: %s" % e)
    else:
        print("The model is valid!")

if __name__ == "__main__":
    args = argparser_onnx_inspector()
    target = args.config_file
    target_path = Path(target)
    if not os.path.exists(target_path):
        print(f"The path {target_path} doesn't exist")
        raise SystemExit(1)

    if os.path.isfile(target_path):
        config = load_config(target_path)
        try:
            onnx_file = config["onnx_file"]
            onnx_path = Path(onnx_file)
            if not os.path.exists(onnx_path):
                print(f"The onnx path {onnx_path} doesn't exist")
                raise SystemExit(1)
            if not os.path.isfile(onnx_path):
                print(f"The onnx path {onnx_path} is not a file")
                raise SystemExit(1)
        except:
            print(f"The onnx file doesn't exist")
            raise SystemExit(1)

        try:
            output_dir = config["output_dir"]
        except:
            output_dir = os.getcwd()
            print(f"Setting the default output directory: {output_dir}")

    main(onnx_path, output_dir)
