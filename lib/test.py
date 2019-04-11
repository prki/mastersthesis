import sys
import img_utils


img_utils.calc_ssim(sys.argv[1], sys.argv[2])
print(img_utils.calc_metrics_luma(sys.argv[1], sys.argv[2]))
