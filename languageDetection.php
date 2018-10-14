<?php
/**
 * Created by PhpStorm.
 * User: felipe
 * Date: 12/10/18
 * Time: 13:53
 */

use Phpml\Classification\NaiveBayes;
use Phpml\CrossValidation\StratifiedRandomSplit;
use Phpml\Dataset\ArrayDataset;
use Phpml\Dataset\CsvDataset;
use Phpml\FeatureExtraction\TfIdfTransformer;
use Phpml\FeatureExtraction\TokenCountVectorizer;
use Phpml\Metric\Accuracy;
use Phpml\Tokenization\WordTokenizer;

include 'vendor/autoload.php';

$dataset = new CsvDataset('datasets/languages.csv', 1);
$vectorizer = new TokenCountVectorizer(new WordTokenizer());
$tfIdTransformer = new TfIdfTransformer();

$samples = [];
foreach ($dataset->getSamples() as $sample) {
    $samples[] = $sample[0];
}

$vectorizer->fit($samples);
$vectorizer->transform($samples);

$tfIdTransformer->fit($samples);
$tfIdTransformer->transform($samples);

$dataset = new ArrayDataset($samples, $dataset->getTargets());
$randomSplit = new StratifiedRandomSplit($dataset, 0.3);

$classifier = new NaiveBayes();
$classifier->train($randomSplit->getTrainSamples(), $randomSplit->getTrainLabels());

$predictLabels = $classifier->predict($randomSplit->getTestSamples());
$accuracy = Accuracy::score($randomSplit->getTestLabels(), $predictLabels);

echo 'Precisão: '.number_format($accuracy*100, 2).'%';