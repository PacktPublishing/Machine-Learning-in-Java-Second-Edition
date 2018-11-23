package chapter3;

import java.util.Random;

import javax.swing.JFrame;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

public class ClassficationWeka {
	public static void main(String args[]) {
		try {
			DataSource source  = new DataSource("data/zoo.arff");
			Instances data = source.getDataSet();
			System.out.println(data.numInstances()+ " instances loaded");
			System.out.println(data.toString());
			
			Remove remove = new Remove();
			String[] opts = new String[] {"-R", "1"};
			remove.setOptions(opts);
			remove.setInputFormat(data);
			data = Filter.useFilter(data, remove);
			System.out.println(data.toString());
			
			InfoGainAttributeEval eval = new InfoGainAttributeEval();
			Ranker search = new Ranker();
			AttributeSelection attselect = new AttributeSelection();
			attselect.setEvaluator(eval);
			attselect.setSearch(search);
			attselect.SelectAttributes(data);
			
			int[] indices = attselect.selectedAttributes();
			System.out.println(Utils.arrayToString(indices));
			
			J48 tree = new J48();
			String[] options = new String[1];
			options[0] = "-U";
			tree.setOptions(options);
			tree.buildClassifier(data);
			System.out.println(tree);
			
			
			double[] vals = new double[data.numAttributes()];
			vals[0] = 1.0; //hair {false, true}
			vals[1] = 0.0; //feathers {false, true}
			vals[2] = 0.0; //eggs {false, true}
			vals[3] = 1.0; //milk {false, true}
			vals[4] = 0.0; //airborne {false, true}
			vals[5] = 0.0; //aquatic {false, true}
			vals[6] = 0.0; //predator {false, true}
			vals[7] = 1.0; //toothed {false, true}
			vals[8] = 1.0; //backbone {false, true}
			vals[9] = 1.0; //breathes {false, true}
			vals[10] = 1.0; //venomous {false, true}
			vals[11] = 0.0; //fins {false, true}
			vals[12] = 4.0; //legs INTEGER [0,9]
			vals[13] = 1.0; //tail {false, true}
			vals[14] = 1.0; //domestic {false, true}
			vals[15] = 0.0; //catsize {false, true}
			Instance myUnicorn = new DenseInstance(1.0, vals);
			
			double result = tree.classifyInstance(myUnicorn);
			System.out.println(data.classAttribute().value((int) result));
			
			Classifier cl = new J48();
			Evaluation eval_roc = new Evaluation(data);
			eval_roc.crossValidateModel(cl, data, 10, new Random(1), new
			Object[] {});
			
			System.out.println(eval_roc.toSummaryString());
			double[][] confusionMatrix = eval_roc.confusionMatrix();
			System.out.println(eval_roc.toMatrixString());
			
		}
		catch(Exception e) {
			System.out.println(e);
			e.printStackTrace();
		}
	}

}
