using ECGPreprocess.Transforms;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Newtonsoft.Json;
using System.IO;

namespace ECGPreprocess.Tests
{
    [TestClass]
    public class Clamp1dTest
    {
        [TestMethod]
        public void Result_Same_As_Torch_Clamp()
        {
            //Arrange
            var sourcePath = @"TestData/source_1000Hz_1s.json";
            var targetPath = @"TestData/result_clamp_-0.3-1.6_1000Hz_1s.json";
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

            var filter = new Clamp1d<double>(-0.3, 1.6);

            //Act
            var result = filter.Transform(source);

            //Assert
            CollectionAssert.AreEqual(target, result);
        }
    }
}
