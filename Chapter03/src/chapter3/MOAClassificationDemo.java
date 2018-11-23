package chapter3;

import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.meta.PairedLearners;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.InstanceExample;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.evaluation.preview.LearningCurve;
import moa.streams.ArffFileStream;
import moa.streams.clustering.SimpleCSVStream;
import moa.tasks.EvaluatePrequential;

public class MOAClassificationDemo {
	public static void main(String args[]) {
		ArffFileStream fs = new ArffFileStream("data/zoo.arff", -1);
		fs.prepareForUse();
		
		PairedLearners learners = new PairedLearners();

		NaiveBayes nb = new NaiveBayes();
		HoeffdingTree ht = new HoeffdingTree();
		
	    BasicClassificationPerformanceEvaluator evaluator = new BasicClassificationPerformanceEvaluator();
	    
	    
	    EvaluatePrequential task = new EvaluatePrequential();
	    task.learnerOption.setCurrentObject(ht);
	    task.streamOption.setCurrentObject(fs);
	    task.evaluatorOption.setCurrentObject(evaluator);

	    task.prepareForUse();

	    LearningCurve le = (LearningCurve) task.doTask();

	    System.out.println(le);
	    
	    
		
	}

}
