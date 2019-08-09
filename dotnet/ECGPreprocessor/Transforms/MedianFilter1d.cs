using ECGPreprocess.Helpers;
using System;

namespace ECGPreprocess.Transforms
{
    public class MedianFilter1d : ITransform1d
    {
        private int kernel_size;

        public MedianFilter1d(int kernel_size)
        {
            if (kernel_size % 2 != 0)
            {
                this.kernel_size = kernel_size;
            }
            else
            {
                throw new ArgumentException("kernel_size must be odd number", "kernel_size");
            }
        } 

        public double[] Transform(double[] x)
        {
            double[] result = new double[x.Length];

            int pad = this.kernel_size / 2;
            double[] x_zero_padded = new double[x.Length + pad * 2];
            Array.Copy(x, 0, x_zero_padded, pad, x.Length);

            double[] sliding_kernel = new double[this.kernel_size];
            for (int i=0; i < x.Length; i++)
            {
                Array.Copy(x_zero_padded, i, sliding_kernel, 0, this.kernel_size);
                result[i] = sliding_kernel.Median();
            }

            return result;
        }
    }
}
