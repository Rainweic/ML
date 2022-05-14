from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper

_UNDEFINED = -1


class BestSplitter:

    def __init__(self, n_bins=3):
        # 分箱处理
        self.n_bins = n_bins
        self.binmapper = _BinMapper(self.n_bins)

    def get_optimal_gini_bid_fid(self, X, y=None):
        """
        计算最佳gini以及其所对应的bid、fid
        分别计算leader方和partner方的 在leader方进行对比得出最佳gini位于何方
        """
        min_gini_local, optimal_bid_local, optimal_fid_local = self._get_min_gini_from_local_data(X, y)

        return min_gini_local, optimal_bid_local, optimal_fid_local

    def data_2_bins(self, X):
        """将数据分箱"""
        if X is not None:
            self.binmapper = self.binmapper.fit(X)
            X = self.binmapper.transform(X)

        return X

    def _cal_ginival_from_local_data(self, sum_label, samples):
        """计算本地数据的gini值"""
        if samples == 0:
            return _UNDEFINED
        prob = sum_label / samples
        return 2 * prob * (1 - prob)

    def _get_min_gini_from_local_data(self, binned_data, label):
        """
        对本地已经分完箱的数据进行计算最小的gini系数
        返回最小的gini系数以及对应的featureID binID
        """
        n_features = binned_data.shape[1]
        n_samples = binned_data.shape[0] if binned_data is not None else label.shape[0]
        gini_bid_fid_list = []
        for fid in range(n_features):

            left_sample = 0
            right_sample = n_samples

            left_label = 0
            right_label = label.sum()

            tmp_binned_data = binned_data[:, fid]

            # 根据分箱进行计算
            for bid in range(self.n_bins - 1):

                # 先选出当前分箱的索引
                mask = (tmp_binned_data == bid)

                # 当前分割点左右两边的样本数量
                bin_sample = mask.sum()
                left_sample += bin_sample
                right_sample -= bin_sample

                # 当前分割点左右两边label为1的样本数量
                bin_sum_label = label[mask].sum()
                left_label += bin_sum_label
                right_label -= bin_sum_label

                # 计算左右两边的gini值
                # 此处由于是2分类 知道其中一类的数量即可知道另一类的数量 故只需要传入一类的数量（label=1）
                left_gini = self._cal_ginival_from_local_data(left_label, left_sample)
                right_gini = self._cal_ginival_from_local_data(right_label, right_sample)

                # 计算gini系数
                if left_gini == _UNDEFINED or right_gini == _UNDEFINED:
                    gini_bid_fid_list.append([1, bid, fid])
                else:
                    gini = (left_gini * left_sample + right_gini * right_sample) / n_samples
                    gini_bid_fid_list.append([gini, bid, fid])

        min_gini, optimal_bid, optimal_fid = min(gini_bid_fid_list, key=lambda x: x[0])

        return min_gini, optimal_bid, optimal_fid

    def get_threshold(self, fid, bid):
        """
        is_global：
            针对参数fid是否对应双方数据集
        """
        # 当bid为最后一个分箱时 其阈值则为最后一个
        if bid == len(self.binmapper.bin_thresholds_[fid]):
            threshold = self.binmapper.bin_thresholds_[fid][-1]
        else:
            try:
                threshold = self.binmapper.bin_thresholds_[fid][bid]
            except BaseException:
                print(f"取阈值超出分箱界限, bin_thresholds_:{self.binmapper.bin_thresholds_} fid:{fid} bid:{bid}")
                raise BaseException
        return threshold

    def split_2_X1_X2(self, record):
        """
        输入record，将其中的X划分为X1，X2两块数据集
        """
        X = record.X
        X_val = record.X_val
        fid = record.fid       
        owner = record.owner
        bid = record.bid

        # 划分训练集
        X1_mask = X[:, fid] <= bid
        X2_mask = X[:, fid] > bid

        # 划分验证集
        if X_val is not None:
            threshold = self.get_threshold(fid, bid, owner, is_global=False)

            X1_val_mask = X_val[:, fid] <= threshold
            X2_val_mask = X_val[:, fid] > threshold

        return X1_mask, X2_mask, X1_val_mask, X2_val_mask
