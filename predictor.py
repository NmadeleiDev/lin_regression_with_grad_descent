import json
import sys
import os
from trainer import fn_to_optimize

def main():
    conf_file_path = './config.json'
    if not os.path.isfile(conf_file_path):
        print(f'Config file must be present at {conf_file_path}')
        return
    with open('./config.json', 'r') as f:
        params = json.load(f)

    try:
        theta1, theta0 = params['theta1'], params['theta0']
    except Exception:
        print("Config file is incorrect")
        return

    while True:
        try:
            km = float(input('Input the mileage...\n'))
        except Exception:
            print("Wrong input, number expected")
            break
        print(f'Predicted price is {fn_to_optimize(km, theta0, theta1)}')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\bBye!')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    except Exception as e:
        print('Program failed:', str(e))