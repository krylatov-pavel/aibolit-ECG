using ECGPreprocess.Helpers;
using System;

namespace ECGPreprocess.Transforms
{
    public class MedianFilter1d<T> : ITransform1d<T> where T: IComparable<T>
    {
        private int kernelSize;

        public MedianFilter1d(int kernelSize)
        {
            if (kernelSize % 2 != 0)
            {
                this.kernelSize = kernelSize;
            }
            else
            {
                throw new ArgumentException("kernel_size must be odd number", "kernel_size");
            }
        } 

        public T[] Transform(T[] x)
        {
            T[] result = new T[x.Length];

            int pad = this.kernelSize / 2;
            var xZeroPadded = new T[x.Length + pad * 2];
            Array.Copy(x, 0, xZeroPadded, pad, x.Length);

            var slidingKernel = new T[this.kernelSize];
            for (var i = 0; i < x.Length; i++)
            {
                Array.Copy(xZeroPadded, i, slidingKernel, 0, this.kernelSize);
                result[i] = slidingKernel.Median();
            }

            return result;
        }
    }
}
