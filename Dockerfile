FROM gcc:13-bookworm AS build
COPY pipe.cpp /src/pipe.cpp
RUN g++ -O3 -mavx2 -mbmi2 -mpopcnt -std=c++17 -pthread -static-libstdc++ -static-libgcc -o /pipe /src/pipe.cpp

FROM debian:bookworm-slim
COPY --from=build /pipe /app/pipe
WORKDIR /app
CMD ["/app/pipe"]
