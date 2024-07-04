
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#include "Matrix.cuh"

int main()
{
	// Matriz con manejo manual
	auto InicioContadorMManu = std::chrono::high_resolution_clock::now();
	MatrixManual MManuA(10000, 10000);
	MatrixManual MManuB(10000, 10000);

	MManuA.LLenarConValor(3.0);
	MManuB.LLenarConValor(2.0);

	MManuA.Sumar(MManuB);
	auto FinContadorMManu = std::chrono::high_resolution_clock::now();

	auto DuracionMManu = std::chrono::duration_cast<std::chrono::milliseconds>(FinContadorMManu - InicioContadorMManu);
	std::cout << "Tiempo transcurrido Matriz manejada manualmente: " << DuracionMManu.count() << " milisegundos\n";

	std::cout << MManuA;
	std::cout << MManuB;

	// Matriz con manejo automatico
	auto InicioContadorMMane = std::chrono::high_resolution_clock::now();
	MatrixManejada MManeA(10000, 10000);
	MatrixManejada MManeB(10000, 10000);

	MManeA.LLenarConValor(3.0);
	MManeB.LLenarConValor(2.0);

	MManeA.Sumar(MManeB);
	auto FinContadorMMane = std::chrono::high_resolution_clock::now();

	auto DuracionMMane = std::chrono::duration_cast<std::chrono::milliseconds>(FinContadorMMane - InicioContadorMMane);
	std::cout << "\n\nTiempo transcurrido Matriz manejada automaticamente: " << DuracionMMane.count() << " milisegundos\n";

	std::cout << MManeA;
	std::cout << MManeB;

	// Matriz con manejo manual y memoria fijada en el host
	auto InicioContadorMFij = std::chrono::high_resolution_clock::now();
	MatrizFijada MFijA(10000, 10000);
	MatrizFijada MFijB(10000, 10000);

	MFijA.LLenarConValor(3.0);
	MFijB.LLenarConValor(2.0);

	MFijA.Sumar(MFijB);
	auto FinContadorMFij = std::chrono::high_resolution_clock::now();

	auto DuracionMFij = std::chrono::duration_cast<std::chrono::milliseconds>(FinContadorMFij - InicioContadorMFij);
	std::cout << "\n\nTiempo transcurrido Matriz fijada: " << DuracionMFij.count() << " milisegundos\n";

	std::cout << MFijA;
	std::cout << MFijB;

	// Matriz con manejo manual, memoria fijada en el host y memoria asincrona en el device
	auto InicioContadorMFijAsync = std::chrono::high_resolution_clock::now();
	MatrizFijada MFijAsyncA(10000, 10000);
	MatrizFijada MFijAsyncB(10000, 10000);

	MFijAsyncA.LLenarConValor(3.0);
	MFijAsyncB.LLenarConValor(2.0);

	MFijAsyncA.Sumar(MFijAsyncB);
	auto FinContadorMFijAsync = std::chrono::high_resolution_clock::now();

	auto DuracionMFijAsync = std::chrono::duration_cast<std::chrono::milliseconds>(FinContadorMFijAsync - InicioContadorMFijAsync);
	std::cout << "\n\nTiempo transcurrido Matriz fijada: " << DuracionMFijAsync.count() << " milisegundos\n";

	std::cout << MFijAsyncA;
	std::cout << MFijAsyncB;
	return 0;
}
