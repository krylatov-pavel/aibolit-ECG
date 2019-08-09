using System;

namespace ECGPreprocess.Transforms
{
    public class Clamp1d<T> : ITransform1d<T> where T: IComparable<T>
    {
        private T min;
        private T max;

        public Clamp1d(T min, T max)
        {
            this.min = min;
            this.max = max;
        }

        public T[] Transform(T[] x)
        {
            var result = new T[x.Length];

            for(var i=0; i < x.Length; i++)
            {
                if (x[i].CompareTo(this.min) < 0)
                {
                    result[i] = this.min;
                }
                else if (x[i].CompareTo(this.max) > 0)
                {
                    result[i] = this.max;
                } else
                {
                    result[i] = x[i];
                }
            }

            return result;
        }
    }
}
