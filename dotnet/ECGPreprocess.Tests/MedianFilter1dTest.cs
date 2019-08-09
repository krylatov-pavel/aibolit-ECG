using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.IO;
using Newtonsoft.Json;
using ECGPreprocess.Transforms;

namespace ECGPreprocess.Tests
{
    [TestClass]
    public class MedianFilter1dTest
    {
        [TestMethod]
        public void Result_Same_As_Scipy_Filter()
        {
            //Arrange
            var sourcePath = @"TestData/source_1000Hz_1s.json";
            var targetPath = @"TestData/result_medfilter_25_1000Hz_1s.json";
            double[] source = null;
            double[] target = null;

            using (StreamReader r = new StreamReader(sourcePath))
            {
                string json = r.ReadToEnd();
                source = JsonConvert.DeserializeObject<double[]>(json);
            }

            using (StreamReader r = new StreamReader(targetPath))
            {
                string json = r.ReadToEnd();
                target = JsonConvert.DeserializeObject<double[]>(json);
            }

            var filter = new MedianFilter1d(25);

            //Act
            var result = filter.Transform(source);

            //Assert
            CollectionAssert.AreEqual(target, result);
        }
    }
}
