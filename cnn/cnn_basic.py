from typing import Literal, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import os


class CNNWithPyTorch(nn.Module):
    # conv_filter_size = 5
    # conv_input = [[3, 6], [6, 16]]
    # conv_stride = 1
    # conv_pool_filter_size = 2
    # conv_pool_stride = 2
    # hidden_node_size = 120
    # output_node_count = 10

    conv2d_list: nn.ModuleList
    pool_list: nn.ModuleList
    fc_list: nn.ModuleList

    def __init__(
        self,
        conv_filter_size: int,
        conv_input: list,
        conv_stride: int,
        conv_pool_filter_size: int,
        conv_pool_stride: int,
        hidden_node_size: int,
        output_node_count: int,
    ):
        super(CNNWithPyTorch, self).__init__()

        self.conv_filter_size = conv_filter_size
        self.conv_input = conv_input
        self.conv_stride = conv_stride
        self.conv_pool_filter_size = conv_pool_filter_size
        self.conv_pool_stride = conv_pool_stride
        self.hidden_node_size = hidden_node_size
        self.output_node_count = output_node_count

        self.conv2d_list = nn.ModuleList()
        self.pool_list = nn.ModuleList()
        self.fc_list = nn.ModuleList()

        for input in self.conv_input:
            self.conv2d_list.append(
                nn.Conv2d(
                    in_channels=input[0],
                    out_channels=input[1],
                    kernel_size=conv_filter_size,
                )
            )
            self.pool_list.append(
                nn.MaxPool2d(
                    kernel_size=self.conv_pool_filter_size, stride=self.conv_pool_stride
                )
            )

        self.fc_list.append(
            nn.Linear(
                in_features=self.conv_input[-1][1]
                * self.conv_filter_size
                * self.conv_filter_size,
                out_features=self.hidden_node_size,
            )
        )
        self.fc_list.append(
            nn.Linear(
                in_features=self.hidden_node_size, out_features=self.output_node_count
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_x: torch.Tensor = x
        for pool, conv in zip(self.pool_list, self.conv2d_list):
            local_x = pool(F.relu(conv(local_x)))

        last_conv_input = self.conv_input[-1]

        local_x = local_x.view(
            -1,
            last_conv_input[1] * self.conv_filter_size * self.conv_filter_size,
        )

        for fc in self.fc_list:
            local_x = F.relu(fc(local_x))

        return local_x

    def train(self, epochs: int, data_loader: torch.utils.data.DataLoader) -> None:
        print("Start training with pytorch")
        criterion = nn.CrossEntropyLoss()  # 손실함수
        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.001, momentum=0.9
        )  # 확률적 경사하강법 : 분류 문제라서 이 방식 추천

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(data_loader, 0):
                inputs, labels = data

                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()  # 초기값들을 0 으로 초기화

                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()  # 역전파
                optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:
                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}")
                    running_loss = 0.0

        print("Finished Training")

        model_save_path = "./cnn/models/cnn_with_pytorch.pth"
        torch.save(self.state_dict(), model_save_path)
        print("Finished Saving Model")

    def train_with_protect_overfit(
        self,
        epochs: int,
        data_loader: torch.utils.data.DataLoader,
    ) -> None:
        print("Start training with overfit protecting code and pytorch")
        criterion = nn.CrossEntropyLoss()  # 손실함수
        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.001, momentum=0.9
        )  # 확률적 경사하강법 : 분류 문제라서 이 방식 추천

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        best_val_loss = float("inf")
        patience = 5  # 조기 중단을 위한 인내심 카운터
        counter = 0

        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(data_loader, 0):
                inputs, labels = data

                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()  # 초기값들을 0 으로 초기화

                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()  # 역전파
                optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:
                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}")
                    running_loss = 0.0

            avg_train_loss = running_loss / len(data_loader)

            # 조기 중단 조건 체크
            if avg_train_loss < best_val_loss:
                best_val_loss = avg_train_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        print("Finished Training")

        model_save_path = "./cnn/models/cnn_with_pytorch.pth"
        torch.save(self.state_dict(), model_save_path)
        print("Finished Saving Model")


class CNNWithTensorFlow:
    def __init__(self):
        pass


def test_with_CIFAR10(framework_type: Literal["pytorch", "tensorflow"]):
    # get CIFAR10 dataset
    # CIFA10 데이터 양식
    # 사이즈 : 32 * 32
    # 타입 : 컬러 이미지
    # 클래스 수 : 10개
    # 클래스 당 이미지 장 수 : 6000장
    # 총 이미지 장 수 : 60000장
    # 학습 이미지 수 : 50000장
    # 테스트 이미지 수 : 10000장

    epochs = 10

    print("data load(CIFAR10) is start")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./datasets/CIFAR10", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True)

    testset = torchvision.datasets.CIFAR10(
        root="./datasets/CIFAR10", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False)

    print("data load(CIFAR10) is end")
    if framework_type == "pytorch":
        print("PyTorch CNN start.")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"device: {device}")

        # 각 변수 별 데이터 지정한 이유
        # conv filter size : 통상적 이유
        # conv input
        #  - 2차원 : 간단하게 실험하기 위해 2개의 레이어만 제공
        #  - 1번째 레이어 : 3 - RGB라서 3차원, 6 - 임의로 아웃풋 숫자 설정
        #  - 2번재 레이어 : 6 - 1번째 레이어 아웃풋 숫자가 6개고 그걸 받는 인풋이라서 동일하게 설정, 16 - 임의로 아웃풋 숫자 설정
        #  - 레이어 파라미터가 2개만 있는 이유 : 원래 레이어 커널(필터) 사이즈도 있어야 하나 위에서 따로 호출하였기에 제외
        # conv stride : 통상적 이유(1칸만 움직이기 위해서)
        # conv pool filter size : 통상적 이유 (2*2 사이즈 커널(필터) 만들기 위해서)
        # hidden node size : 임의로 120개 설정
        # output node count : 데이터 셋의 클래스 종류가 10종이라서

        # 추후 테스트 해 볼 리스트
        # 레이어 커널 사이즈, 레이어 아웃풋 사이즈 등 통상적 혹은 임의로 설정한 데이터를 변경하면서 어떤 차이가 있는지 테스트
        # 레이어 갯수 자체에 대한 테스트도 해보면 좋을 듯 함

        cnn_model = CNNWithPyTorch(
            conv_filter_size=5,
            conv_input=[[3, 6], [6, 16]],
            conv_stride=1,
            conv_pool_filter_size=2,
            conv_pool_stride=2,
            hidden_node_size=120,
            output_node_count=10,
        ).to(device)
        print(f"load cnn with pytorch is end. model : {cnn_model}")

        # cnn_model.train(epochs=epochs, data_loader=trainloader)
        cnn_model.train_with_protect_overfit(epochs=epochs, data_loader=trainloader)

        # load model
        cnn_model.load_state_dict(
            torch.load(os.path.join("cnn", "models", "cnn_with_pytorch.pth"))
        )

        correct = 0
        total = 0

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = cnn_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(
            f"Accuracy of the network on the 10000 test images: {100 * correct / total}"
        )
        print(f"correct : {correct}, total : {total}")
        print(f"Outputs data : {outputs.data}")
        print(f"Predicted data : {predicted}")
        print("end training with pytorch")

        # epochs 10 : Accuracy of the network on the 10000 test images: 64.7
        # epochs 20 : Accuracy of the network on the 10000 test images: 62.87
        # epochs 100 :Accuracy of the network on the 10000 test images: 56.95
        # epochs 300 :Accuracy of the network on the 10000 test images: 53.71
    elif framework_type == "tensorflow":
        pass


print("end test_with_CIFAR10")


if __name__ == "__main__":
    print("start cnn running")
    framework_type: str = "pytorch"
    test_with_CIFAR10(framework_type=framework_type)
