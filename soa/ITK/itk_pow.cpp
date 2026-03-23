#include "itkImageRegistrationMethodv4.h"
#include "itkEuler3DTransform.h"
#include "itkJointHistogramMutualInformationImageToImageMetricv4.h"
#include "itkPowellOptimizerv4.h"
#include "itkNormalVariateGenerator.h"
#include "itkImageSeriesReader.h"
#include "itkImageSeriesWriter.h"
#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkNumericSeriesFileNames.h"
#include "itkPNGImageIO.h"
#include "itkExtractImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkRegistrationParameterScalesFromPhysicalShift.h"
#include "itkStatisticsImageFilter.h"
#include "itkCenteredTransformInitializer.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImage.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionConstIterator.h"
#include "utils.hpp"

#include <filesystem>
#include <algorithm>
#include <iostream>
#include <stdexcept>

#include <vector>
#include "itkCommand.h"

struct RotationAngles {
    double x; 
    double y;    
    double z; 
};

RotationAngles getRotationAnglesFromMatrix(const itk::Matrix<double, 3, 3>& rotationMatrix) {
    RotationAngles angles;

    angles.z = std::atan2(rotationMatrix(1, 0), rotationMatrix(0, 0));

    angles.y = std::atan2(-rotationMatrix(2, 0), std::sqrt(rotationMatrix(2, 1) * rotationMatrix(2, 1) + rotationMatrix(2, 2) * rotationMatrix(2, 2)));

    angles.x = std::atan2(rotationMatrix(2, 1), rotationMatrix(2, 2));

    return angles;
}

class CommandIterationUpdate : public itk::Command
{
public:
  using Self = CommandIterationUpdate;
  using Superclass = itk::Command;
  using Pointer = itk::SmartPointer<Self>;
  itkNewMacro(Self);

protected:
  CommandIterationUpdate() { m_LastMetricValue = 0.0; };

public:
  using OptimizerType = itk::PowellOptimizerv4<double>;
  using OptimizerPointer = const OptimizerType *;

  void
  Execute(itk::Object * caller, const itk::EventObject & event) override
  {
    Execute((const itk::Object *)caller, event);
  }

  void
  Execute(const itk::Object * object, const itk::EventObject & event) override
  {
    auto optimizer = static_cast<OptimizerPointer>(object);
    if (!itk::IterationEvent().CheckEvent(&event))
    {
      return;
    }
    double currentValue = optimizer->GetValue();
    if (itk::Math::abs(m_LastMetricValue - currentValue) > 1e-7)
    {

      m_LastMetricValue = currentValue;
    }
  }

private:
  double m_LastMetricValue;
};

int main(int argc, char * argv[])
{
    if (argc < 4)
    {
    std::cerr << "Missing Parameters " << std::endl;
    std::cerr << "Usage: " << argv[0];
    std::cerr << " ct_path pet_path outputImageFile [numberOfIterations] [samplingPercentage] " << std::endl;
    return EXIT_FAILURE;
    }

    constexpr unsigned int Dimension = 3;
    using PixelType = float;
    using FixedImageType = itk::Image<PixelType, Dimension>;
    using FixedImageType2D = itk::Image<PixelType, 2>;

    using MovingImageType = itk::Image<PixelType, Dimension>;
    using OutputImageType = itk::Image<uint8_t, Dimension>;
    using OutputImageType2D = itk::Image<uint8_t, 2>;

    using TransformType = itk::Euler3DTransform<double>;
    using InterpolatorType = itk::LinearInterpolateImageFunction<MovingImageType, double>;
    using OptimizerType = itk::PowellOptimizerv4<double>;
    using RegistrationType = itk::ImageRegistrationMethodv4<FixedImageType, MovingImageType>;
    using MetricType = itk::JointHistogramMutualInformationImageToImageMetricv4<FixedImageType, MovingImageType>;

    using ExtractFilterType = itk::ExtractImageFilter<OutputImageType, OutputImageType2D>;
    using WriterType = itk::ImageFileWriter<OutputImageType2D>;

    typename FixedImageType::Pointer fixedImage;
    typename MovingImageType::Pointer movingImage;

    try
    {
        ReadImageSeriesPNG<FixedImageType>(argv[1], fixedImage);
        ReadImageSeriesPNG<MovingImageType>(argv[2], movingImage);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << "Error during image reading: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }

    try
    {
        RescaleAndWriteImages<MovingImageType, OutputImageType, OutputImageType2D>(fixedImage, argv[3], "fixedImage");
        RescaleAndWriteImages<MovingImageType, OutputImageType, OutputImageType2D>(movingImage, argv[3], "movingImage");

    }
    catch (itk::ExceptionObject &error)
    {
        std::cerr << "Error during rescaling and writing images: " << error << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Images read and written successfully" << std::endl;

    auto transform = TransformType::New(); // Euler3DTransform
    auto optimizer = OptimizerType::New(); // PowellOptimizerv4
    auto metric = MetricType::New(); // JointHistogramMutualInformationImageToImageMetricv4
    auto registration = RegistrationType::New(); // ImageRegistrationMethodv4
    auto interpolator = InterpolatorType::New(); // LinearInterpolateImageFunction
    
    metric->SetVarianceForJointPDFSmoothing(0);
    metric->SetNumberOfHistogramBins(256);
    metric->SetFixedInterpolator(interpolator);
    metric->SetMovingInterpolator(interpolator);
    registration->SetMetric(metric);

    double samplingPercentage = 1.0;
    if (argc > 4)
    {
    samplingPercentage = std::stod(argv[4]);
    }
    registration->SetMetricSamplingPercentage(samplingPercentage);
    registration->SetMetricSamplingStrategy(RegistrationType::MetricSamplingStrategyEnum::RANDOM);
    registration->SetFixedImage(fixedImage);
    registration->SetMovingImage(movingImage);
    std::cout << "Metric Set Finished, now setting transform" << std::endl;

    std::cout << "Compute Transform Centers" << std::endl;
    std::cout << "Initialize Transform With Moments" << std::endl;
    
    using TransformInitializerType = itk::CenteredTransformInitializer<itk::Euler3DTransform<double>, FixedImageType, MovingImageType>;
    auto initializer = TransformInitializerType::New();
    initializer->SetTransform(transform);
    initializer->SetFixedImage(fixedImage);
    initializer->SetMovingImage(movingImage);
    initializer->MomentsOn();  
    initializer->InitializeTransform();
    
    TransformType::MatrixType matrix = transform->GetMatrix();
    TransformType::OffsetType offset = transform->GetOffset();
    
    offset[2] = 0;  // Set Z translation to 0
    
    transform->SetOffset(offset);
    
    std::cout << "Matrix = " << std::endl << matrix << std::endl;
    std::cout << "Offset = " << std::endl << offset << std::endl;
    
    registration->SetInitialTransform(transform);

    optimizer->SetStepLength(argc > 5 ? std::stod(argv[5]) : 0.0005);
    optimizer->SetStepTolerance(argc > 6 ? std::stod(argv[6]) : 0.0005);
    optimizer->SetValueTolerance(argc > 7 ? std::stod(argv[7]) : 0.00001);
    optimizer->SetMaximumIteration(argc > 8 ? std::stoi(argv[8]) : 200);
    optimizer->SetMetric(metric);
    constexpr unsigned int numberOfLevels = 1;
    using ScalesEstimatorType = itk::RegistrationParameterScalesFromPhysicalShift<MetricType>;
    ScalesEstimatorType::Pointer scalesEstimator = ScalesEstimatorType::New();
    scalesEstimator->SetMetric(metric);
    scalesEstimator->SetTransformForward(true); 
    scalesEstimator->SetSmallParameterVariation(1);
    optimizer->SetScalesEstimator(scalesEstimator);

    auto observer = CommandIterationUpdate::New();
    optimizer->AddObserver(itk::IterationEvent(), observer);

    registration -> SetOptimizer(optimizer);

    RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
    shrinkFactorsPerLevel.SetSize(1);
    shrinkFactorsPerLevel[0] = 1;

    RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
    smoothingSigmasPerLevel.SetSize(1);
    smoothingSigmasPerLevel[0] = 0;

    registration->SetNumberOfLevels(numberOfLevels);
    registration->SetSmoothingSigmasPerLevel(smoothingSigmasPerLevel);
    registration->SetShrinkFactorsPerLevel(shrinkFactorsPerLevel);

    try
    {
    std::cout << "Starting registration" << std::endl;  
    auto start = std::chrono::high_resolution_clock::now();
    registration->Update();
    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << "Registration completed!" << std::endl;
    std::cout << "Optimizer stop condition: "
              << registration->GetOptimizer()->GetStopConditionDescription()
              << std::endl;

    std::chrono::duration<double> elapsed = finish - start;

    TransformType::ParametersType finalParameters = registration->GetOutput()->Get()->GetParameters();
    
    double       centerX = finalParameters[0];
    double       centerY = finalParameters[1];
    double       centerZ = finalParameters[2];
    double       finalTranslationX = finalParameters[3];
    double       finalTranslationY = finalParameters[4];
    double       finalTranslationZ = finalParameters[5];
    unsigned int numberOfIterations = optimizer->GetCurrentIteration();
    double       bestValue = optimizer->GetValue();
    finalParameters[5] = 0;

    auto finalTransform = TransformType::New();
    finalTransform->SetParameters(finalParameters);
    TransformType::MatrixType matrix2 = finalTransform->GetMatrix();
    TransformType::OffsetType offset2 = finalTransform->GetOffset();
    auto angles = getRotationAnglesFromMatrix(matrix2);

    using ResampleFilterType = itk::ResampleImageFilter<MovingImageType, FixedImageType>;
    auto resampler = ResampleFilterType::New();
    resampler->SetInput(movingImage);          
    resampler->SetSize(fixedImage->GetLargestPossibleRegion().GetSize());  
    resampler->SetOutputSpacing(fixedImage->GetSpacing());               
    resampler->SetOutputOrigin(fixedImage->GetOrigin());                 
    resampler->SetSize(fixedImage->GetLargestPossibleRegion().GetSize()); 
    
    resampler->SetTransform(finalTransform); 

    auto interpolator = itk::LinearInterpolateImageFunction<MovingImageType, double>::New();
    resampler->SetInterpolator(interpolator);
    
    typename MovingImageType::Pointer outputImage = resampler->GetOutput();

    std::chrono::duration<double> elapsed_resample = std::chrono::high_resolution_clock::now() - start;
    std::ofstream file;
    file.open("itk_times.csv", std::ios_base::app);
    file << elapsed.count() << "," << elapsed_resample.count() << std::endl;
    file.close();

    RescaleAndWriteImages<MovingImageType, OutputImageType, OutputImageType2D>(outputImage, argv[3], "outputImage");
    return EXIT_SUCCESS;
    }catch (itk::ExceptionObject & ex)
      {
        std::cerr << "Exception caught during registration: " << ex << std::endl;
        return EXIT_FAILURE;
      }

}