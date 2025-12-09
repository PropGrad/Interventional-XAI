import numpy as np
import bmi
import matplotlib.pyplot as plt

class Zollstock:
    """
    Class to measure and visualize changes
    in prediction behavior, along a feature axis.

    This is our main class for measuring impact.
    It includes various modes for measuring changes in predictions:
    - "grad": average absolute gradient
    - "corr": Pearson correlation coefficient
    - "mi": mutual information

    We also provide methods for performing a shuffle test
    to assess significance of measured changes, as well
    as some visualization methods.
    """
    def __init__(self, mode="grad"):
        self.mode = mode 

        if mode == "grad": # average absolute gradient
            self.test_statistic = self.mean_abs_grad
        elif mode == "corr": # pearson correlation coefficient
            self.test_statistic =  self.corr_coeff
        elif mode == "mi": # mutual information
            self.test_statistic = self.mut_inf
        else:
            raise ValueError(f"mode {mode} is not implemented!")


    def measure_avg_gradient(self, predictions):
        predictions = np.array(predictions)
        assert len(predictions.shape) == 1, "only 1d predictions are expected currently"

        grads = np.gradient(predictions)
        return np.mean(grads)
    

    def prediction_flips(self, predictions, return_num=False):
        predictions = np.array(predictions)
        predictions = np.argmax(predictions, axis=1)

        if return_num:
            flips = 0
            for i in range(1, len(predictions)):
                if predictions[i] != predictions[i-1]:
                    flips += 1
            return flips
        else:
            return len(np.unique(predictions)) > 1
        
    
    def mean_abs_grad(self, predictions):
        predictions = np.array(predictions)
        assert len(predictions.shape) == 1, "only 1d predictions are expected currently"

        grads = np.gradient(predictions)
        return np.mean(np.abs(grads))
    
    def corr_coeff(self, predictions):
        x = np.linspace(0,1,len(predictions), endpoint=True)
        rho = np.corrcoef(predictions, x)[0,1]
        return rho
    
    def mut_inf(self, predictions):
        x = np.linspace(0,1,len(predictions), endpoint=True)

        ksg = bmi.estimators.KSGEnsembleFirstEstimator(neighborhoods=(5,))
        mi = ksg.estimate(predictions.reshape(-1, 1), x.reshape(-1, 1))
        return mi
    

    def calc_p_val(self, base, permutations):
        if self.mode == "grad":
            # shift it to zero mean
            m_perm = np.mean(permutations)
            corrected_perm = permutations - m_perm
            corrected_base = base - m_perm
            # now calculate two sided p value
            # here larger than is important to catch constant output as not significant
            return np.sum(np.abs(corrected_perm) >= np.abs(corrected_base))/len(permutations)
        elif self.mode == "corr":
            return np.sum(np.abs(permutations) >= np.abs(base))/len(permutations)
        elif self.mode == "mi":
            return np.sum(permutations > base)/len(permutations)
        else:
            raise ValueError(f"mode {self.mode} is not implemented!")

    def shuffle_plot(self, preds, num=5,
                     figsize=(8,5), 
                     true_color="#f72585",
                     shuffle_color="#4895ef"):
        """Plot shuffled values vs. original values"""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        for i in range(num):
            perm = np.random.permutation(preds)
            label = None
            if i == num-1:
                label = "Permutations"

            ax.plot(perm, label=label, color=shuffle_color, alpha=0.5,
                    linestyle="--", linewidth=1)

        ax.plot(preds, label="Original Observation", color=true_color, linestyle="-", linewidth=3)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
                    ncol=2, fancybox=True, shadow=True)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.tight_layout()

        return fig

    def nuldist_plot(self, null, original,
                     figsize=(8,5),
                     true_color="#f72585",
                     null_color="#4895ef"):
        """Plot estimated null distribution"""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ax.hist(null, bins=100, label="Null Distribution", color=null_color, alpha=0.8)
        ax.axvline(original, color=true_color, linewidth=3, linestyle="--", label="Original Observation")

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
                    ncol=2, fancybox=True, shadow=True)
        
        ax.set_xlabel("Measured Statistic")
        ax.set_ylabel("Number of Occurances")
        fig.tight_layout()

        return fig

    
    def shuffle_test(self, predictions, num=1000, plot_null_dist=False, plot_shuffle=False):
        preds = np.array(predictions).copy()
        assert len(preds.shape) == 1, "only 1d predictions are expected currently"

        statistic = self.test_statistic(preds)        
        permuted = []
        for i in range(num):
            perm = np.random.permutation(preds)
            permuted.append(self.test_statistic(perm))

            if i == 0 and plot_shuffle:
                self.shuffle_plot(preds)

        permuted = np.array(permuted)
        
        null_plot = None
        if plot_null_dist:
            null_plot = self.nuldist_plot(permuted, statistic)

        if null_plot:
            return statistic, self.calc_p_val(statistic, permuted), null_plot
        else:
            return statistic, self.calc_p_val(statistic, permuted)