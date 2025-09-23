# viz/base.py

import os
import matplotlib.pyplot as plt

class StrideViz:
    """A base class for handling visualization and saving of plots."""

    def __init__(self, save_dir: str | None = None, dpi: int = 120, show: bool = False):
        """Initializes the StrideViz instance.

        Args:
            save_dir (str | None): The directory to save plots. If None, plots are not saved.
            dpi (int): The resolution for saved figures.
            show (bool): Whether to display plots using plt.show().
        """
        self.save_dir = save_dir
        self.dpi = dpi
        self.show = show
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    def _finalize(self, fig, title: str, fname: str | None):
        """Finalizes and saves or shows a matplotlib figure.

        This is an internal helper method to apply consistent styling and handling.

        Args:
            fig: The matplotlib figure object.
            title (str): The super title for the figure.
            fname (str | None): The filename for saving the figure.

        Returns:
            str | None: The path to the saved file, or None.
        """
        fig.suptitle(title, y=0.99, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = None
        if self.save_dir and fname:
            out_path = os.path.join(self.save_dir, fname)
            fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        if self.show:
            plt.show()
        else:
            plt.close(fig)
        return out_path