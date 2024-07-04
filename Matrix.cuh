#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <chrono>

__global__ void SumarKernel(int CantidadElementos, float* DatosPrimeraMatriz, float* DatosSegundaMatriz)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < CantidadElementos)
	{
		DatosPrimeraMatriz[i] = DatosPrimeraMatriz[i] + DatosSegundaMatriz[i];
	}
}

class MatrixManual
{
public:
	int NumeroFilas;
	int NumeroColumnas;
	int CantidadElementos;

	float* ContenedorDatosHost;
	float* ContenedorDatosDevice;

	MatrixManual(int ArgNumeroFilas, int ArgNumeroColumnas)
	{
		NumeroFilas = ArgNumeroFilas;
		NumeroColumnas = ArgNumeroColumnas;
		CantidadElementos = NumeroFilas * NumeroColumnas;
		ContenedorDatosHost = new float[CantidadElementos];
		for (int i = 0; i < CantidadElementos; i++)
		{
			ContenedorDatosHost[i] = 0.0;
		}
		cudaMalloc(&ContenedorDatosDevice, sizeof(float) * CantidadElementos);
		cudaMemcpy(ContenedorDatosDevice, ContenedorDatosHost, sizeof(float) * CantidadElementos, cudaMemcpyHostToDevice);
	}

	~MatrixManual()
	{
		cudaFree(ContenedorDatosDevice);
	}

	friend std::ostream& operator<<(std::ostream& Stream, MatrixManual& Matriz)
	{
		Stream << "\nImprimiendo 10 valores para comprobar que funcione la matriz: ";
		for (int i = 0; i < 10; i++)
		{
			Stream << Matriz.ContenedorDatosHost[i] << ",";
		}
		return Stream;
	}

	void LLenarConValor(float Valor)
	{
		for (int i = 0; i < CantidadElementos; i++)
		{
			ContenedorDatosHost[i] = Valor;
		}
		cudaMemcpy(ContenedorDatosDevice, ContenedorDatosHost, sizeof(float) * CantidadElementos, cudaMemcpyHostToDevice);
	}

	void Sumar(MatrixManual SegundaMatrix)
	{
		int BlockSize = 256;
		int NumBlocks = (CantidadElementos + BlockSize - 1) / BlockSize;

		SumarKernel << <NumBlocks, BlockSize >> > (CantidadElementos, ContenedorDatosDevice, SegundaMatrix.ContenedorDatosDevice);
		cudaMemcpy(ContenedorDatosHost, ContenedorDatosDevice, CantidadElementos * sizeof(float), cudaMemcpyDeviceToHost);
	}
};

class MatrixManejada
{
public:
	int NumeroFilas;
	int NumeroColumnas;
	int CantidadElementos;

	float* ContenedorDatos;

	MatrixManejada(int ArgNumeroFilas, int ArgNumeroColumnas)
	{
		NumeroFilas = ArgNumeroFilas;
		NumeroColumnas = ArgNumeroColumnas;
		CantidadElementos = NumeroFilas * NumeroColumnas;
		cudaMallocManaged(&ContenedorDatos, sizeof(float) * CantidadElementos);
		for (int i = 0; i < CantidadElementos; i++)
		{
			ContenedorDatos[i] = 0.0;
		}
		cudaDeviceSynchronize();
	}

	//~MatrixManejada()
	//{
	//	cudaFree(ContenedorDatos);
	//}

	friend std::ostream& operator<<(std::ostream& Stream, MatrixManejada& Matriz)
	{
		cudaDeviceSynchronize();
		Stream << "\nImprimiendo 10 valores para comprobar que funcione la matriz: ";
		for (int i = 0; i < 10; i++)
		{
			Stream << Matriz.ContenedorDatos[i] << ",";
		}
		return Stream;
	}

	void LLenarConValor(float Valor)
	{
		for (int i = 0; i < CantidadElementos; i++)
		{
			ContenedorDatos[i] = Valor;
		}
		cudaDeviceSynchronize();
	}

	void Sumar(MatrixManejada SegundaMatrix)
	{
		int BlockSize = 256;
		int NumBlocks = (CantidadElementos + BlockSize - 1) / BlockSize;

		cudaDeviceSynchronize();
		SumarKernel << <NumBlocks, BlockSize >> > (CantidadElementos, ContenedorDatos, SegundaMatrix.ContenedorDatos);
		cudaDeviceSynchronize();
	}
};

class MatrizFijada
{

public:
	int NumeroFilas;
	int NumeroColumnas;
	int CantidadElementos;

	float* ContenedorDatosHost;
	float* ContenedorDatosDevice;

	MatrizFijada(int ArgNumeroFilas, int ArgNumeroColumnas)
	{
		NumeroFilas = ArgNumeroFilas;
		NumeroColumnas = ArgNumeroColumnas;
		CantidadElementos = NumeroFilas * NumeroColumnas;
		cudaMallocHost(&ContenedorDatosHost, sizeof(float) * CantidadElementos);
		for (int i = 0; i < CantidadElementos; i++)
		{
			ContenedorDatosHost[i] = 0.0;
		}
		cudaMalloc(&ContenedorDatosDevice, sizeof(float) * CantidadElementos);
		cudaMemcpy(ContenedorDatosDevice, ContenedorDatosHost, sizeof(float) * CantidadElementos, cudaMemcpyHostToDevice);
	}

	~MatrizFijada()
	{
		cudaFree(ContenedorDatosDevice);
	}

	friend std::ostream& operator<<(std::ostream& Stream, MatrizFijada& Matriz)
	{
		Stream << "\nImprimiendo 10 valores para comprobar que funcione la matriz: ";
		for (int i = 0; i < 10; i++)
		{
			Stream << Matriz.ContenedorDatosHost[i] << ",";
		}
		return Stream;
	}

	void LLenarConValor(float Valor)
	{
		for (int i = 0; i < CantidadElementos; i++)
		{
			ContenedorDatosHost[i] = Valor;
		}
		cudaMemcpy(ContenedorDatosDevice, ContenedorDatosHost, sizeof(float) * CantidadElementos, cudaMemcpyHostToDevice);
	}

	void Sumar(MatrizFijada SegundaMatrix)
	{
		int BlockSize = 256;
		int NumBlocks = (CantidadElementos + BlockSize - 1) / BlockSize;

		SumarKernel << <NumBlocks, BlockSize >> > (CantidadElementos, ContenedorDatosDevice, SegundaMatrix.ContenedorDatosDevice);
		cudaMemcpy(ContenedorDatosHost, ContenedorDatosDevice, CantidadElementos * sizeof(float), cudaMemcpyDeviceToHost);
	}
};


class MatrizFijadaAsync
{

public:
	int NumeroFilas;
	int NumeroColumnas;
	int CantidadElementos;

	float* ContenedorDatosHost;
	float* ContenedorDatosDevice;

	MatrizFijadaAsync(int ArgNumeroFilas, int ArgNumeroColumnas)
	{
		NumeroFilas = ArgNumeroFilas;
		NumeroColumnas = ArgNumeroColumnas;
		CantidadElementos = NumeroFilas * NumeroColumnas;
		cudaMallocHost(&ContenedorDatosHost, sizeof(float) * CantidadElementos);
		for (int i = 0; i < CantidadElementos; i++)
		{
			ContenedorDatosHost[i] = 0.0;
		}
		cudaMallocAsync(&ContenedorDatosDevice, sizeof(float) * CantidadElementos, cudaStreamDefault);
		cudaMemcpyAsync(ContenedorDatosDevice, ContenedorDatosHost, sizeof(float) * CantidadElementos, cudaMemcpyHostToDevice, cudaStreamDefault);
	}

	~MatrizFijadaAsync()
	{
		cudaFreeAsync(ContenedorDatosDevice, cudaStreamDefault);
	}

	friend std::ostream& operator<<(std::ostream& Stream, MatrizFijadaAsync& Matriz)
	{
		Stream << "\nImprimiendo 10 valores para comprobar que funcione la matriz: ";
		for (int i = 0; i < 10; i++)
		{
			Stream << Matriz.ContenedorDatosHost[i] << ",";
		}
		return Stream;
	}

	void LLenarConValor(float Valor)
	{
		for (int i = 0; i < CantidadElementos; i++)
		{
			ContenedorDatosHost[i] = Valor;
		}
		cudaMemcpyAsync(ContenedorDatosDevice, ContenedorDatosHost, sizeof(float) * CantidadElementos, cudaMemcpyHostToDevice, cudaStreamDefault);
	}

	void Sumar(MatrizFijadaAsync SegundaMatrix)
	{
		int BlockSize = 256;
		int NumBlocks = (CantidadElementos + BlockSize - 1) / BlockSize;

		SumarKernel << <NumBlocks, BlockSize >> > (CantidadElementos, ContenedorDatosDevice, SegundaMatrix.ContenedorDatosDevice);
		cudaMemcpyAsync(ContenedorDatosHost, ContenedorDatosDevice, CantidadElementos * sizeof(float), cudaMemcpyDeviceToHost, cudaStreamDefault);
	}
};