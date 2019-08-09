using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.IO;
using Newtonsoft.Json;
using ECGPreprocess.Preprocessors;

namespace ECGPreprocess.Tests
{
    [TestClass]
    public class DensenetPreprocessorTest
    {
        [TestMethod]
        public void Result_Same_As_Python_Preprocessor()
        {
            //preprocessor includes:
            //median filter, window 25
            //clamp -19 .. 20
            //scale 0 .. 5

            //Arrange
            var sourcePath = @"TestData/source_1000Hz_1s.json";
            var targetPath = @"TestData/result_densenet_preprocessor_1000Hz_1s.json";
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

            var transform = new DensenetPreprocessor();

            //Act
            var result = transform.Transform(source);

            //Assert
            CollectionAssert.AreEqual(target, result);
        }
    }
}
