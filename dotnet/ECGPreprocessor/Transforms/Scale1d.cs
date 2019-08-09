namespace ECGPreprocess.Transforms
{
    public class Scale1d : ITransform1d<double>
    {
        private double sourceMin;
        private double sourceMax;
        private double targetMin;
        private double targetMax;

        public Scale1d(double sourceMin, double sourceMax, double targetMin, double targetMax)
        {
            this.sourceMin = sourceMin;
            this.sourceMax = sourceMax;
            this.targetMin = targetMin;
            this.targetMax = targetMax;
        }

        public double[] Transform(double[] x)
        {
            var result = new double[x.Length];

            for (var i = 0; i < x.Length; i++)
            {
                result[i] = ((this.targetMax - this.targetMin) * (x[i] - this.sourceMin) / (this.sourceMax - this.sourceMin)) + this.targetMin;
            }

            return result;
        }
    }
}
