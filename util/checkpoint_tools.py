import os
from tensorflow.keras.callbacks import Callback


class CheckpointTools(Callback):
    def __init__(self, cp_dir, save_best_only=True, num_saves=3):
        self.cp_dir = cp_dir
        self.last_val_loss = float("inf")    # save_best_only判定用
        self.save_best_only = save_best_only
        assert num_saves >= 1
        self.num_saves = num_saves    # 最大保存数(この数を超えたら最古を消す)
        self.recent_files = []        # ファイル履歴

    def remove_oldest_file(self):
        if len(self.recent_files) > self.num_saves:
            file_name = self.recent_files.pop(0)  # 先頭ファイルパス取得
            if os.path.exists(file_name):
                os.remove(file_name)              # ファイル削除
            print('remove:'+file_name)

    # 毎epoch 終了時に呼び出されるCallback
    def on_epoch_end(self, epoch, logs={}):
        val_loss = logs['val_loss']

        # ModelCheckpointのファイル名に合わせる
        # ※epoch=(epoch+1) に注意
        file_name = os.path.join(self.cp_dir, 'epoch{epoch:03d}-{val_loss:.5f}.hdf5').format(epoch=(epoch+1), val_loss=val_loss)
        print('store:'+file_name)

        if self.save_best_only:
            if val_loss < self.last_val_loss:
                self.last_val_loss = val_loss
                self.recent_files.append(file_name)
                self.remove_oldest_file()
        else:
            # ファイル履歴追加
            self.recent_files.append(file_name)
            # 古いファイル削除
            self.remove_oldest_file()