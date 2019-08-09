using System;

namespace ECGPreprocess.Transforms
{
    public interface ITransform1d<T> where T: IComparable<T>
    {
        T[] Transform(T[] x);
    }
}
