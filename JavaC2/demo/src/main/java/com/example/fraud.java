package com.example;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

import javax.swing.*;
import java.awt.*;
import java.util.Arrays;

public class fraud {

    public static void main(String[] args) {
        try {
            // Load dataset
            DataSource source = new DataSource("C:\\TRISEM III\\JAVA\\Rayan\\demo\\src\\main\\java\\com\\example\\ccfraudj.csv");
            Instances data = source.getDataSet();

            // Set class attribute
            data.setClassIndex(data.numAttributes() - 1);

            // Perform exploratory analysis
            performExploratoryAnalysis(data);

            // Build and evaluate classifier
            Classifier classifier = new J48(); // You can choose any other classifier
            classifier.buildClassifier(data);

            // Visualize decision tree
            visualizeTree(classifier);

            // Evaluate classifier using cross-validation
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(classifier, data, 10, new java.util.Random(1));
            System.out.println(eval.toSummaryString("\nResults\n======\n", false));
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void performExploratoryAnalysis(Instances data) {
        // Exploratory analysis
        System.out.println("Exploratory Analysis:");
        System.out.println("Number of instances: " + data.numInstances());
        System.out.println("Number of attributes: " + data.numAttributes());
        System.out.println("**************************");
        System.out.println("Attributes: ");
        for (int i = 0; i < data.numAttributes(); i++) {
            System.out.println(data.attribute(i));
        }
        System.out.println("**************************");

        System.out.println("Class attribute: " + data.attribute(data.classIndex()));
        System.out.println("Class distribution:");
        System.out.println(data.attributeStats(data.classIndex()));
        System.out.println("**************************");
        // Summary statistics
        System.out.println("\nSummary Statistics:");
        for (int i = 0; i < data.numAttributes(); i++) {
            System.out.println("Attribute: " + data.attribute(i).name());
            System.out.println("Type: " + data.attribute(i).type());
            if (data.attribute(i).isNumeric()) {
                System.out.println("Mean: " + data.attributeStats(i).numericStats.mean);
                System.out.println("Std. Deviation: " + data.attributeStats(i).numericStats.stdDev);
                System.out.println("Minimum: " + data.attributeStats(i).numericStats.min);
                System.out.println("Maximum: " + data.attributeStats(i).numericStats.max);
                System.out.println("*************************");
            }
        }

        // Correlation Analysis
        System.out.println("\nCorrelation Analysis:");
        double[][] correlationMatrix = calculateCorrelationMatrix(data);
        printCorrelationMatrix(correlationMatrix);
    }

    private static double[][] calculateCorrelationMatrix(Instances data) {
        int numAttributes = data.numAttributes();
        double[][] correlationMatrix = new double[numAttributes][numAttributes];

        for (int i = 0; i < numAttributes; i++) {
            for (int j = i; j < numAttributes; j++) {
                correlationMatrix[i][j] = calculateCorrelation(data.attributeToDoubleArray(i), data.attributeToDoubleArray(j));
                correlationMatrix[j][i] = correlationMatrix[i][j];
            }
        }

        return correlationMatrix;
    }

    private static double calculateCorrelation(double[] x, double[] y) {
        double sumX = 0.0, sumY = 0.0, sumXY = 0.0, sumX2 = 0.0, sumY2 = 0.0;

        for (int i = 0; i < x.length; i++) {
            sumX += x[i];
            sumY += y[i];
            sumXY += x[i] * y[i];
            sumX2 += x[i] * x[i];
            sumY2 += y[i] * y[i];
        }

        int n = x.length;
        return (n * sumXY - sumX * sumY) / Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    }

    private static void printCorrelationMatrix(double[][] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                System.out.print(matrix[i][j] + "\t");
            }
            System.out.println();
        }
    }

    private static void visualizeTree(Classifier classifier) throws Exception {
        // Create tree visualizer
        TreeVisualizer treeVisualizer = new TreeVisualizer(null, ((J48) classifier).graph(), new PlaceNode2());

        // Display tree
        JFrame jFrame = new JFrame("Decision Tree Visualizer");
        jFrame.setSize(1800, 1200);
        jFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        jFrame.getContentPane().setLayout(new BorderLayout());
        jFrame.getContentPane().add(treeVisualizer, BorderLayout.CENTER);
        jFrame.setVisible(true);

        // Start visualization
        treeVisualizer.fitToScreen();
    }
}