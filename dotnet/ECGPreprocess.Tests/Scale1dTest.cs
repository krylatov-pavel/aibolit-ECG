using ECGPreprocess.Transforms;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Newtonsoft.Json;
using System.IO;

namespace ECGPreprocess.Tests
{
    [TestClass]
    public class Scale1dTest
    {
        [TestMethod]
        public void Result_Same_As_Python_Scale()
        { 
            //Arrange
            var sourcePath = @"TestData/source_1000Hz_1s.json";
            var targetPath = @"TestData/result_scale_-0.3-1.6_0-1_1000Hz_1s.json";
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

            var transform = new Scale1d(-0.3, 1.6, 0, 1);

            //Act
            var result = transform.Transform(source);

            //Assert
            CollectionAssert.AreEqual(target, result);
        }
    }
}
