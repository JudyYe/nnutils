import pickle

def unpickle_exec():
    import sys
    files = sys.argv[1]
    with open(files, 'rb') as fp:
        obj = pickle.load(fp)
    func = obj['func']
    func_args = obj['args']
    func(func_args)
    

if __name__ == '__main__':
    unpickle_exec()