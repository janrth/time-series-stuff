import numpy as np

class DataPreProcessing(object):
    
    def __init__(self):
        pass
    
    def split_sequence_for_id(self, df, actions, target, n_steps=None, autoregressive=None):
        '''
            autoregressive --> uses the target as feature and creates X,y that y is index+1 compared to x array
            n_steps --> number of input steps used
            actions --> list of features. If no features used it will use the target as shifted feature using autoregressive mode

        '''
        X, y = list(), list()

        for acc_id in df.account_id.unique():
            df_temp = df[df.account_id==acc_id]
            arr = df_temp[actions].T.values
            arr_target = df_temp[target].T.values

            if len(actions)>0:

                arr_new = []
                for i in range(arr.shape[0]):
                    if i ==0:
                        arr_one = arr[i].reshape((arr.shape[1], 1))
                        arr_new.append(arr_one)
                    else:
                        arr_temp = arr[i].reshape((arr.shape[1], 1))
                        arr_new.append(arr_temp)

                # horizontally stack columns      
                out_seq = arr_target.reshape((arr_target.shape[0], 1))
                seq_temp = np.hstack(arr_new)
                if autoregressive:
                    sequences = np.hstack((seq_temp, out_seq, out_seq))
                else:
                    sequences = np.hstack((seq_temp, out_seq))
            else:
                out_seq = arr_target.reshape((arr_target.shape[0], 1))
                sequences = np.hstack(out_seq)

            if n_steps:
                n_steps = n_steps
            else:
                n_steps = len(sequences)


            for i in range(len(sequences)):
                # find the end of this pattern
                end_ix = i + n_steps

                if len(actions)<=0:
                    # check if we are beyond the dataset
                    if end_ix > len(sequences)-1:
                        break
                    # gather input and output parts of the pattern
                    seq_x, seq_y = sequences[i:end_ix], sequences[end_ix]
                else:
                    if autoregressive:
                        # check if we are beyond the dataset
                        if end_ix > len(sequences)-1:
                            break
                        # gather input and output parts of the pattern
                        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix, -1]
                    else:
                        # check if we are beyond the dataset
                        if end_ix > len(sequences):
                            break
                        # gather input and output parts of the pattern
                        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]


                X.append(seq_x)
                y.append(seq_y)
        return np.array(X), np.array(y)