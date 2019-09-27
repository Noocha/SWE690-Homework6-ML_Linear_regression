import java.io.File;
import java.io.IOException;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class LinearRegressionML {
    public static void main(String[] args) {
        process("src/Web_site_visitors_2014-2019_training.arff", "src/Web_site_visitors_2014-2019_testing.arff", "src/Web_site_visitors_2014-2019_predict.arff");
    }

    public static Instances getDataSet(String fileName) {
        try {
            ArffLoader loader = new ArffLoader();
            loader.setFile(new File(fileName));
            Instances dataSet = loader.getDataSet();
            dataSet.setClassIndex(4);
            return dataSet;

        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static void process(String trainFileName, String testFileName, String predictFileName) {
        try {
            Instances trainingDataSet = getDataSet(trainFileName);
            Instances testingDataSet = getDataSet(testFileName);

            //choose type of supervised learning to be Linear Regression
            Classifier classifier = new LinearRegression();
            classifier.buildClassifier(trainingDataSet);

            Evaluation eval = new Evaluation(trainingDataSet);
            eval.evaluateModel(classifier, testingDataSet);

            System.out.println("Linear Regression");
            System.out.println(eval.toSummaryString());
            System.out.println("Expression for the input data");
            System.out.println(classifier);

            //Predicting
            System.out.println("Prediction");
            Instance predictingDataSet = getDataSet(predictFileName).lastInstance();
            double value = classifier.classifyInstance(predictingDataSet);
            System.out.println("Predict value of Returning visitor is ");
            System.out.println(value);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
