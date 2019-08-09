using ECGPreprocess.Transforms;

namespace ECGPreprocess.Preprocessors
{
    public class DensenetPreprocessor : ITransform1d<double>
    {
        private ITransform1d<double>[] transforms;

        public DensenetPreprocessor()
        {
            this.transforms = new ITransform1d<double>[]
            {
                new MedianFilter1d<double>(25),
                new Clamp1d<double>(-19.0, 21.0),
                new Scale1d(-19.0, 21.0, 0, 5.0)
            };
        }

        public double[] Transform(double[] x)
        {
            var result = x;

            for(var i = 0; i < this.transforms.Length; i++)
            {
                result = this.transforms[i].Transform(result);
            }

            return result;
        }
    }
}
