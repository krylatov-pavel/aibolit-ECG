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
                throw new ArgumentException("kernel_size must be odd number", "kernelSize");
            }
        } 

        public T[] Transform(T[] x)
        {
            T[] result = new T[x.Length];

            int pad = this.kernelSize / 2;

            var slidingKernel = new T[this.kernelSize];
            for (var i = 0; i < x.Length; i++)
            {
                Array.Clear(slidingKernel, 0, slidingKernel.Length);

                //zero pad sliding kernel on edges
                var sourceIdx = Math.Max(i - pad, 0);
                var destinationIdx = Math.Max(pad - i, 0);
                var length = this.kernelSize - destinationIdx - Math.Max(pad - (x.Length - 1 - i), 0);
                Array.Copy(x, sourceIdx, slidingKernel, destinationIdx, length);

                result[i] = slidingKernel.Median();
            }

            return result;
        }
    }
}
