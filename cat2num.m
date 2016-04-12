function out=cat2num(r)
%%% it will convert the contegories to numericals. It assumes that data is
%%% in cell format
%%%%each row is a feature vector starting from the first row.
[R,C]=size(r);
for i=1:C
    temp1=cell2mat(r(1,i));
    if(isnumeric(temp1))
        %num_f(j)=i;%%%%numerical features
        %j=j+1;
       out(:,i)=cell2mat(r(:,i)); 
        
    else
        %cat_f(k)=i;
        [a,b,c]=(unique(r(:,i)));
        out(:,i)=c;
        %k=k+1;
    end
end