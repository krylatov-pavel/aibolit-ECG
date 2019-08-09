using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ECGPreprocess.Helpers
{
    public static class ArrayExtension
    {
        public static T Median<T>(this T[] array) where T: IComparable<T>
        {
            if (array.Length % 2 == 1)
            {
                int medianIdx = array.Length / 2;

                return QuickSelect<T>(array, medianIdx);
            }
            else
            {
                throw new NotImplementedException("Median calculation of even-length arrays not implemented");
            }
        }

        private static T QuickSelect<T>(T[] array, int k) where T : IComparable<T>
        {

            if (array.Length == 1)
            {
                if (k == 0)
                {
                    return array[0];
                }
                else
                {
                    throw new ArgumentOutOfRangeException("Something's wrong with algorithm implementation");
                }
            }
            else
            {
                Random rnd = new Random();
                T pivot = array[rnd.Next(array.Length)];

                IEnumerable<T> less = array.Where(e => e.CompareTo(pivot) < 0);
                IEnumerable<T> pivots = array.Where(e => e.Equals(pivot));
                IEnumerable<T> greater = array.Where(e => e.CompareTo(pivot) > 0);

                if (k < less.Count())
                {
                    return QuickSelect(less.ToArray(), k);
                }
                if (k < less.Count() + pivots.Count())
                {
                    return pivots.First();
                }
                else
                {
                    return QuickSelect(greater.ToArray(), k - less.Count() - pivots.Count());
                }
            }

        }
    }
}
