using System;

namespace ECGPreprocess.Helpers
{
    public static class ArrayExtension
    {
        public static T Median<T>(this T[] array) where T: IComparable<T>
        {
            if (array.Length % 2 == 1)
            {
                int medianIdx = array.Length / 2;
                var arrayCopy = (T[])array.Clone();
                                    
                return QuickSelect(arrayCopy, 0, arrayCopy.Length - 1, medianIdx);
            }
            else
            {
                throw new NotImplementedException("Median calculation of even-length arrays not implemented");
            }
        }

        private static void Swap<T>(T[] array, int i, int j)
        {
            var buf = array[i];
            array[i] = array[j];
            array[j] = buf;
        }

        private static int Partition<T>(T[] array, int left, int right, int pivotIdx) where T : IComparable<T>
        {
            var pivot = array[pivotIdx];
            Swap(array, pivotIdx, right);

            pivotIdx = left;

            for(var i = left; i < right; i++)
            {
                if (array[i].CompareTo(pivot) <= 0)
                {
                    Swap(array, i, pivotIdx);
                    pivotIdx++;
                }
            }

            Swap(array, pivotIdx, right);

            return pivotIdx;
        }

        private static T QuickSelect<T>(T[] array, int left, int right, int k) where T : IComparable<T>
        {
            if (left == right)
            {
                return array[left];
            }

            var pivotIdx = new Random().Next(left, right);
            pivotIdx = Partition(array, left, right, pivotIdx);

            if (pivotIdx == k)
            {
                return array[pivotIdx];
            }
            else if (pivotIdx > k)
            {
                return QuickSelect(array, left, pivotIdx - 1, k);
            }
            else
            {
                return QuickSelect(array, pivotIdx + 1, right, k);
            }

        }
    }
}
